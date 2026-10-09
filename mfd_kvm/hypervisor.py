# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: MIT
"""Module for KVM."""

import configparser
import logging
import re
import typing
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path, PurePosixPath
from random import choice
from time import sleep, time
from typing import Dict, List, Optional, Tuple, Union, OrderedDict
from uuid import uuid4

from jinja2 import Environment, select_autoescape
from mfd_common_libs import log_levels, add_logging_level, TimeoutCounter, os_supported
from mfd_connect.util.rpc_copy_utils import copy
from mfd_typing import MACAddress, PCIAddress, OSName
from netaddr import IPAddress

from mfd_kvm.data_structures import VFDetail
from mfd_kvm.exceptions import (
    KVMHypervisorExecutionException,
    KVMHypervisorException,
    VFExceptionKVM,
    NotFoundInterfaceKVM,
)
from mfd_kvm.virsh import VirshInterface

if typing.TYPE_CHECKING:
    from uuid import UUID
    from mfd_connect import PythonConnection

logger = logging.getLogger(__name__)
add_logging_level(level_name="MODULE_DEBUG", level_value=log_levels.MODULE_DEBUG)


@dataclass
class VMParams:
    """
    Configuration for VM.

    name: Name of VM
    is_uefi_mode: Determine UEFI Boot mode
    cpu_count: Count of CPUs
    memory: Memory value in MB
    mac_address: Mac address of main adapter
    machine: Emulation machine, eg. pc or q35, chipset
    bridge_name: Name of bridge interface
    clone_disk: Determine whether disk image should be copied or not
    threads: Determine number of threads, If value is omitted, the rest will be autofilled preferring sockets over
            cores over threads.
    disk: Disk of VM, default, disk will be not created, size of disk(in GB) or path
    target_hd_clone_disk: Target location for disk images clones
    os_variant: Future OS, for optimization VM by HV
    boot_order: Order for booting process (in qemu format, eg. 'network,hd') docs: virt-install --boot=BOOTOPTS
    clone_timeout: Timeout for cloning disk image, if used disk
    mdev_uuid: Mediated device ID
    graphics: Specifies the graphical display configuration. Default '--graphics none' will be set.
              Check --graphics in virt-install for available options
    cpu: CPU model and CPU features exposed to the guest
    osinfo_detect: Specifies whether virt-install should attempt OS detection
    osinfo_require: When this option is set to True, virt-install will raise error if no OS detected.
                    If none of these osinfo params are set and creation of VM fails we will set detect=True
                    require=False and try again
    add_pci_controller: When set this option and is_uefi_mode to true then pci controllers will be added to the VM,
                    by default set to False
    installation_method: Installation method, one of the following: --location, --cdroom, --pxe, --import
    disk_bus: disk bus type, one of the following: sata, scsi, virtio, usb, default: None
    arch: vm architecture, one of the following: aarch64, default: None
    vm_xml_file: xml file with VM definition to use instead virt-install: virsh define vm_xml_file
    """

    name: str = "vm"
    is_uefi_mode: bool = True
    cpu_count: int = 2
    memory: int = 1024
    mac_address: MACAddress = MACAddress("00:00:00:00:00:00")
    machine: str = "pc"
    bridge_name: str = "br0"
    clone_disk: bool = True
    threads: Optional[int] = None
    disk: Optional[str] = None
    target_hd_clone_disk: Optional[Path] = None
    os_variant: Optional[str] = None
    boot_order: Optional[str] = None
    clone_timeout: Optional[int] = None
    graphics: Optional[str] = None
    cpu: Optional[str] = None
    osinfo_detect: Optional[bool] = None
    osinfo_require: Optional[bool] = None
    add_pci_controller: Optional[bool] = False
    installation_method: Optional[str] = None
    disk_bus: Optional[str] = None
    arch: Optional[str] = None
    vm_xml_file: Optional[str] = None

    def __post_init__(self):
        for field in fields(self):
            # skip typing.Any, typing.Union, typing.ClassVar without parameters
            if isinstance(field.type, typing._SpecialForm):
                continue
            value = getattr(self, field.name)
            # check if typing.Any, typing.Union, typing.ClassVar with parameters
            try:
                actual_type = field.type.__origin__
            except AttributeError:
                # primitive type
                actual_type = field.type
            # typing.Any, typing.Union, typing.ClassVar
            if isinstance(actual_type, typing._SpecialForm):
                actual_type = field.type.__args__
                if len(actual_type) > 2 or not value:
                    # None value, or multiple args in Union
                    continue
                if type(None) in actual_type:
                    actual_type = actual_type[0]
            if actual_type is Path and value is not None:
                setattr(self, field.name, PurePosixPath(value))
                continue


class KVMHypervisor:
    """Class for kvm hypervisor."""

    # pattern for single line data of vm, separated by 3 spaces,
    # excluding header of list (Id Name State), used for findall multiline
    # eg.
    #  Id    Name                           State
    # ----------------------------------------------------
    #  1     foo-055-045                  running
    LIST_PATTERN = re.compile(r"^\s*(?P<id>(?!Id)\d+|-)\s{2,}(?P<name>\S*)\s+(?P<state>.*)$", re.M)
    #  /sys/class/net/eth2/device/virtfn3 -> ../0000:5e:02.3 -> 0000:5e:02.3
    VF_PCI_ADDRESS_REGEX = r".*(?P<address>\w{4}:\w{2}:\w{2}.\d{1})"
    UUID_REGEX = r"\S{8}-\S{4}-\S{4}-\S{4}-\S{12}"
    DEFAULT_TIMEOUT = 1000
    PCI_CONTROLLER_XML_TMP = "pci_controller_template.xml"

    @os_supported(OSName.LINUX)
    def __init__(self, *, connection: "PythonConnection") -> None:
        """
        Initialize hypervisor.

        :param connection: Python connection
        """
        self._conn = connection
        self.virt_tool = VirshInterface(connection=connection)

    @cached_property
    def _rsync_available(self) -> bool:
        """Check once per hypervisor instance whether rsync is available."""
        return (
            self._conn.execute_command(
                "command -v rsync",
                shell=True,
                expected_return_codes={0, 1},
            ).return_code
            == 0
        )

    @staticmethod
    def get_name_from_ip(ip: IPAddress, prefix: str = "amval") -> str:
        """
        Convert IP into VM name with pattern "foo-<3rd octet of IP>-<4th octet of IP>".

        eg. foo-010-102

        :param ip: Source of conversion
        :param prefix: Prefix of name, default foo
        :return Generated name of VM
        """
        # words - tuple of octets
        return f"{prefix}-{ip.words[2]:03d}-{ip.words[3]:03d}"

    def get_free_network_data(self, *, config_file: Path, count: int) -> List[Tuple[IPAddress, MACAddress]]:
        """
        Get 'count' IPs and MAC addresses of free (not-used) vm guest.

        Procedure of selecting vm-guest is as follows:
        1. get the pool of available IP addresses from config_file
        2. select randomly IP from the pool
        3. check if host with selected IP is running (using ping)
        4. if yes - back to step 2
        5. if not - this host is not used. Select it. Return IP and MAC
        6. Repeat steps 2-4 max. 3 times

        :param config_file: Path to config with network info
        :param count: Count of pairs IP/MAC
        :raise KVMHypervisorException: on failure
        :return - List of tuples (IPAddress, MACAddress)
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Trying to find {count} IP/MAC pair/s.")
        try:
            config_data = self.parse_network_data_conf(config_path=config_file)
        except KVMHypervisorException:
            raise KVMHypervisorException("VM's IP configuration is empty!")
        pairs = []
        # three attempts to find the free guest
        for _ in range(0, 3):
            operational_data = config_data[:]
            while operational_data and len(pairs) < count:
                # select randomly IP address
                read_data = choice(operational_data)
                operational_data.remove(read_data)
                if read_data in pairs:
                    # already used IP
                    continue
                ip, mac = read_data
                result = self._conn.execute_command(f"ping {ip} -c 5", discard_stdout=True, expected_return_codes=None)
                if result.return_code != 0:
                    logger.log(log_levels.MODULE_DEBUG, msg=f"IP {ip} is available.")
                    pairs.append((ip, mac))
                else:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"IP {ip} responded on ping. Trying next one.",
                    )
            sleep(5)
        if pairs:
            return pairs
        raise KVMHypervisorException("Not found expected count of free IPs.")

    @staticmethod
    def parse_network_data_conf(
        config_path: Path,
    ) -> List[Tuple[IPAddress, MACAddress]]:
        """
        Get list of IP addresses and MACs pair defined.

        File contains IP addresses and MACs pair available for VM guests for each config
        Lines started with # are comments and are not considered.

        Example:
        -------
            [kvm]
            10.150.4.160
            #10.10.10.11 <- this IP will be rejected
            10.10.10.12
        :param config_path: Path to file with IPs

        :return: List of available IPs/MAC Addresses
        :raise KVMHypervisorException: when can't get IPs
        """
        raw_config = configparser.RawConfigParser(allow_no_value=True, delimiters="=")
        ips_list = []
        with open(config_path) as config_file:
            raw_config.read_file(config_file)
            if raw_config.has_section("kvm"):
                for data in raw_config.options("kvm"):
                    if data.startswith("#"):
                        continue
                    ip, mac = data.split()
                    ips_list.append((IPAddress(ip), MACAddress(mac)))
        if ips_list:
            return ips_list
        else:
            raise KVMHypervisorException(f"Cannot parse config {config_file}")

    def create_vm(self, params: VMParams) -> str:  # noqa C901
        """
        Create VM from parameters.

        :param params: Parameters for VM creating
        :raise KVMHypervisorExecutionException: if command failed try one more time with the addition of
                                                --osinfo detect=on,require=off (if osinfo not provided in first try),
                                                if this also fails exception will be raised
        :raises FileNotFoundError: if not found disk path
        :raises KVMHypervisorException: if disk cloning failed or timeout occurred
        :return: Name of started VM
        """
        used_hdd = False
        command = [
            "virt-install",
            f"--name={params.name}",
            f"--memory={params.memory}",
            f"--vcpus={params.cpu_count}"
            if params.threads is None
            else f"--vcpus={params.cpu_count},threads={params.threads}",
            f"--machine={params.machine}",
            "--noautoconsole",
        ]
        if int(params.cpu_count) > 255:
            command.append("--iommu model=intel,driver.intremap=on,driver.eim=on,driver.caching_mode=on")
            command.append("--features apic=on,ioapic.driver=qemu")

        if params.mac_address is not None:
            command.append(f"--network=bridge:{params.bridge_name},mac={params.mac_address},model=virtio")
        if params.os_variant is not None:
            command.append(f"--os-variant={params.os_variant}")
        if params.disk is None:
            command.append("--disk=none")
        else:
            try:
                size = int(params.disk)
                command.append(f"--disk size={size}")
            except ValueError:
                used_hdd = True
                path_to_source_image = self._conn.path(params.disk)

                disk_args = []
                if params.clone_disk:
                    clone_timeout = params.clone_timeout if params.clone_timeout else self.DEFAULT_TIMEOUT

                    path_to_destination_image = (
                        self._conn.path(
                            params.target_hd_clone_disk if params.target_hd_clone_disk else path_to_source_image.parent
                        )
                        / params.name
                    )
                    cloned_disk_path = self.clone_vm_hdd_image(
                        path_to_source_image=path_to_source_image,
                        path_to_destination_image=path_to_destination_image,
                        timeout=clone_timeout,
                    )
                    disk_args.append(f"path={cloned_disk_path}")
                    sleep(5)  # lack of sleep between image copying and vm creation might lead to problems with swtpm
                else:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Using {path_to_source_image} image for new VM. - 'clone_disk' parameter set to False.",
                    )
                    disk_args.append(f"path={path_to_source_image}")

                if params.disk_bus is not None:
                    disk_args.append(f"bus={params.disk_bus}")

                command.append(f"--disk {','.join(disk_args)}")

        if params.arch is not None:
            command.append(f"--arch {params.arch}")
        if params.boot_order is not None:
            boot_string = f"--boot={params.boot_order}"
        elif used_hdd:
            boot_string = "--boot=hd"
        else:
            boot_string = "--boot=network,hd"
        if params.is_uefi_mode:
            boot_string += ",uefi"
        command.append(boot_string)

        if params.graphics is not None:
            command.append(f"--graphics {params.graphics}")
        else:
            command.append("--graphics none")

        if params.cpu is not None:
            command.append(f"--cpu={params.cpu}")

        osinfo_args = []
        if params.osinfo_detect is not None:
            osinfo_args.append(f"detect={'on' if params.osinfo_detect else 'off'}")
        if params.osinfo_require is not None:
            osinfo_args.append(f"require={'on' if params.osinfo_require else 'off'}")
        if osinfo_args:
            command.append(f"--osinfo {','.join(osinfo_args)}")

        if params.installation_method is not None:
            command.append(params.installation_method)

        logger.log(log_levels.MODULE_DEBUG, msg="Creating VM")
        # change list into str command
        command = " ".join(command)
        try:
            self._conn.execute_command(command, custom_exception=KVMHypervisorExecutionException)
        except KVMHypervisorExecutionException as kvm_error:
            if "--osinfo" in kvm_error.stderr and params.osinfo_detect is None and params.osinfo_require is None:
                logger.log(
                    log_levels.MODULE_DEBUG,
                    msg="--osinfo problem detected, trying to create vm "
                    "one more time with --osinfo detect=on,require=off."
                    "Problem may be caused by new fatal error added in 2022, "
                    "when require not set to False and OS detection fails."
                    "Setting disk params from source path to just cloned VM path.",
                )
                params.osinfo_detect = True
                params.osinfo_require = False
                if params.disk is not None:
                    params.clone_disk = False
                    params.disk = str(cloned_disk_path)
                self.create_vm(params)
            elif "--import" in kvm_error.stderr and params.installation_method is None:
                logger.log(
                    log_levels.MODULE_DEBUG,
                    msg="Installation method problem detected. Create VM one more time with --import",
                )
                params.installation_method = "--import"
                self.create_vm(params)
            else:
                raise kvm_error

        if params.is_uefi_mode and params.add_pci_controller:
            xml = self.dump_xml(name=params.name)
            last_entry = xml.findall(".//controller[@model='pcie-root-port']")[-1]
            f_index = int(last_entry.get("index")) + 1
            f_chassis = int(last_entry.find("target").get("chassis")) + 1
            f_port = int(last_entry.find("target").get("port"), 16) + 1
            f_bus = int(last_entry.find("address").get("bus"), 16)
            f_slot = int(last_entry.find("address").get("slot"), 16)
            f_func = int(last_entry.find("address").get("function"), 16) + 1

            self.attach_pci_controllers(
                name=params.name,
                number_of_devices=64,
                first_index=f_index,
                first_chassis=f_chassis,
                first_port=f_port,
                first_bus=f_bus,
                first_slot=f_slot,
                first_func=f_func,
            )

        return params.name

    def create_multiple_vms(
        self,
        *,
        count: int = 2,
        params: VMParams,
        ip_data_config_file: Path,
        prefix: str = "amval",
    ) -> List[Tuple[str, IPAddress]]:
        """
        Create multiple VMs with the same configuration.

        :param count: Count of VMs
        :param params: Parameters for VMs configs
        :param ip_data_config_file: Config with IP/MAC data for getting free IP
        :param prefix: Prefix of name, default amval
        :return: List of created VMs (names, ips)
        """
        list_of_vms = []
        network_data = self.get_free_network_data(config_file=ip_data_config_file, count=count)
        for i in range(count):
            ip, mac = network_data[i]
            logger.log(log_levels.MODULE_DEBUG, msg=f"Preparing configuration for {i + 1} VM.")
            params.mac_address = mac
            params.name = self.get_name_from_ip(ip, prefix=prefix)
            list_of_vms.append((self.create_vm(params), ip))
        return list_of_vms

    def get_list_of_vms(self) -> List[str]:
        """
        Get list of hypervisor's VMs names.

        :return: List of names
        """
        vms = []
        result = self.list_vms(True)
        for vm in result:
            vms.append(vm["name"].rstrip())
        return vms

    def get_vm_status(self, name: str) -> Dict[str, str]:
        """
        Get status of a VM.

        :param name: Name of a VM
        :return: Info about VM
        :raises: VirshException if virt_tool fails.
        """
        return self.virt_tool.get_vm_status(name)

    def shutdown_gracefully_vm(self, name: str) -> None:
        """
        Gracefully shutdown VM.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.shutdown_gracefully_vm(name)

    def reboot_vm(self, name: str) -> None:
        """
        Gracefully reboot VM.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.reboot_vm(name)

    def reset_vm(self, name: str) -> None:
        """
        Hard reset VM.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.reset_vm(name)

    def shutdown_vm(self, name: str) -> None:
        """
        Hard shutdown VM.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.shutdown_vm(name)

    def start_vm(self, name: str) -> None:
        """
        Start VM.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.start_vm(name)

    def delete_vm(self, name: str) -> None:
        """
        Delete VM.

        VM must be shutdown.

        :param name: Name of VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.delete_vm(name)

    def wait_for_vm_state(self, name: str, state: str, timeout: int = 60) -> bool:
        """Wait for state on VM for timeout time.

        :param name: Name of VM
        :param state: state for which to wait for
        :param timeout: seconds to wait, default 60 sec
        :return: True if state present, False when timeout
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Waiting for state {state} on {name}...")
        start = time()
        found = False
        while not found and (time() - start < timeout):
            try:
                if state.lower() == self.get_vm_status(name)["State"].lower():
                    found = True
                    break
            except AttributeError:
                pass
            sleep(5)
        if found:
            logger.log(log_levels.MODULE_DEBUG, msg=f"Desired state set: {state}")
        return found

    def wait_for_vm_down(self, vm_id: str, timeout: int = 120) -> bool:
        """
        Wait for down state on vm_id virtual machine for timeout time.

        :param vm_id: virtual machine name/id
        :param timeout: seconds to wait, default 120 secs
        :return: True if vm down, False when timeout
        """
        return self.wait_for_vm_state(vm_id, "shut off", timeout)

    def wait_for_vm_up(self, name: str, timeout: int = 120) -> bool:
        """
        Wait for UP state on vm_id virtual machine for timeout time.

        :param name: virtual machine name/id
        :param timeout: seconds to wait, default 120 secs
        :return: True if vm down, False when timeout
        """
        return self.wait_for_vm_state(name, "running", timeout)

    def stop_all_vms(self, force: bool = False) -> bool:
        """
        Stop all VMs on host.

        :param force: Force to stop vm.
        :return: True if all VMs stop, otherwise False
        """
        vms = self.list_vms()
        ret_code = True
        for vm in vms:
            if force:
                self.shutdown_vm(vm["name"])
            else:
                self.shutdown_gracefully_vm(vm["name"])
            ret = self.wait_for_vm_down(vm["name"])
            if not ret:
                ret_code = False
        return ret_code

    def start_all_vms(self) -> bool:
        """
        Start all VMs on host.

        :return: True if all VMs start, otherwise False
        """
        vms = self.list_vms(True)
        for vm in vms:
            self.start_vm(vm["name"])
            ret = self.wait_for_vm_up(vm["name"])
            if not ret:
                return False
        return True

    def get_vfs_id_for_pf(self, interface: str) -> List[int]:
        """
        Get VFs for PF (interface) using /sys/class/net/interface_name.

        :param interface: PF interface name that has VFs.
        :raises VFExceptionKVM: If VFs are not found.
        :raises KVMHypervisorExecutionException: If command fails.
        :return: List of VF numbers.
        """
        result = self._conn.execute_command(
            f"ls /sys/class/net/{interface}/device/virtfn* -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        vf_number_regex = r"^.*device/virtfn(?P<vf_number>\d+).*$"
        match = re.findall(vf_number_regex, result.stdout, re.M)
        if match:
            return [int(vf) for vf in match]
        else:
            raise VFExceptionKVM(f"Not matched vfs for interface {interface}.")

    def get_pci_address_for_vf(self, *, interface: str, vf_id: int) -> PCIAddress:
        """
        Get pci address of VF using /sys/class/net/interface_name.

        :param interface: PF interface name that has VFs.
        :param vf_id: VF ID number.
        :raises VFExceptionKVM: If pci address is not found.
        :raises KVMHypervisorExecutionException: If command fails.
        :return: PCI Address of VF.
        """
        result = self._conn.execute_command(
            f"ls /sys/class/net/{interface}/device/virtfn{vf_id} -la",
            custom_exception=KVMHypervisorExecutionException,
        )
        match = re.match(self.VF_PCI_ADDRESS_REGEX, result.stdout)
        if match:
            pci_address_data = [int(data, 16) for data in re.split(r"\W+", match.group("address"))]
            return PCIAddress(*pci_address_data)
        else:
            raise VFExceptionKVM(f"Not matched vf {vf_id} for interface {interface}.")

    def get_pci_address_for_vf_by_pci(self, pf_pci_address: PCIAddress, vf_id: int) -> PCIAddress:
        """
        Get pci address of VF using /sys/bus/pci/devices/pci_address.

        :param pf_pci_address: PF interface PCI address.
        :param vf_id: VF ID to find pci address.
        :raises VFExceptionKVM: If pci address is not found.
        :raises KVMHypervisorExecutionException: If command failed.
        :return: PCI Address of VF.
        """
        result = self._conn.execute_command(
            f"ls /sys/bus/pci/devices/{pf_pci_address}/virtfn{vf_id} -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        match = re.match(self.VF_PCI_ADDRESS_REGEX, result.stdout)
        if match:
            pci_address_data = [int(data, 16) for data in re.split(r"\W+", match.group("address"))]
            return PCIAddress(*pci_address_data)
        else:
            raise VFExceptionKVM(f"Not matched vf {vf_id} for interface with PCI address:{pf_pci_address}.")

    def _split_lspci_for_devices(self) -> Dict:
        """
        Split dictionary of devices out of lspci output.

        :return: dictionary of devices
        """
        output = self._conn.execute_command("lspci -k", expected_return_codes={0}).stdout
        lspci = {}
        pci = ""
        for line in output.splitlines():
            if not line[0].isspace():
                pci = line.split()[0]
                lspci[pci] = line
            else:
                lspci[pci] = lspci[pci] + "\n" + line
        return lspci

    def is_vf_attached(self, *, interface: str, vf_id: int) -> bool:
        """
        Determine whether VF interface is attached to a VM (pci passthrough) or not.

        :param interface: PF name
        :param vf_id: VF number
        :return: True if attached, False otherwise
        """
        output = self._split_lspci_for_devices()
        vf_pci = self.get_pci_address_for_vf(interface=interface, vf_id=vf_id)
        vf_pci_format = vf_pci.lspci if vf_pci.lspci in output else vf_pci.lspci_short
        if vf_pci_format not in output:
            raise KVMHypervisorException(
                f"VF PCI: {vf_pci_format} is missing in `lspci -k` output. Cannot check VF attaching state."
            )
        return bool(re.search(r"Kernel\sdriver\sin\suse:\svfio-pci", output[str(vf_pci_format)]))

    def set_number_of_vfs_for_pf(
        self, *, interface: str, vfs_count: int, check: bool = True, timeout: int = 60
    ) -> None:
        """
        Assign VFs for PF (interface) using /sys/class/net/interface_name/device/sriov_numvfs.

        :param interface: PF interface name
        :param vfs_count: What to set sriov_numvfs/how many VFs to set.
        :param check: When True check if created number of VFs matches required number of VFs.
        :param timeout: Max time to execute (secs).
        :raises NotFoundInterfaceKVM: If interface is incorrect
        :raises VFExceptionKVM: If count of VFs is different from expected.
        :raises KVMHypervisorExecutionException: If command failed.
        :raises TimeoutExpired: If timeout occurred.
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Configuring {vfs_count} VFs for {interface}.")
        # check if interface exists
        self._conn.execute_command(f"ls /sys/class/net/{interface}", custom_exception=NotFoundInterfaceKVM)
        result = self._conn.execute_command(
            f"echo {vfs_count} > /sys/class/net/{interface}/device/sriov_numvfs",
            shell=True,
            expected_return_codes=[0, 1],
            custom_exception=KVMHypervisorExecutionException,
            timeout=timeout,
        )
        busy_message = "echo: write error: Device or resource busy"
        # if VFs are defined, need to disable sriov
        if result.return_code == 1 and busy_message in result.stderr:
            logger.log(
                log_levels.MODULE_DEBUG,
                msg="VFs are already configured, disabling and trying again.",
            )
            self._conn.execute_command(
                f"echo 0 > /sys/class/net/{interface}/device/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=timeout,
            )
            self._conn.execute_command(
                f"echo {vfs_count} > /sys/class/net/{interface}/device/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=timeout,
            )
        if check:
            self.check_number_of_vfs(interface=interface, vfs_count=vfs_count)

    def check_number_of_vfs(self, *, interface: str, vfs_count: int) -> None:
        """
        Check if number of vfs is correct using /sys/class/net/interface_name.

        :param interface: PF interface name.
        :param vfs_count: Expected number of VFs.
        :raise VFExceptionKVM: If count of VFs is different from expected.
        :raises KVMHypervisorExecutionException: If command failed.
        """
        result = self._conn.execute_command(
            f"ls /sys/class/net/{interface}/device/virtfn* -la",
            shell=True,
            expected_return_codes={0, 2},
            custom_exception=KVMHypervisorExecutionException,
        )
        if "No such file or directory" in result.stderr:  # no VFs were created
            created_vfs_count = 0
        else:
            vf_regex = r"^.*device/virtfn(?P<vf_number>\d+).*$"
            match = [m.groupdict() for m in re.finditer(vf_regex, result.stdout, re.M)]
            created_vfs_count = len(match)

        if created_vfs_count != vfs_count:
            raise VFExceptionKVM(
                f"Mismatched count of created and expected VFs. Created: {created_vfs_count}, expected: {vfs_count}"
            )
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Correctly assigned {vfs_count} for PF {interface}.",
        )

    def prepare_vf_xml(
        self,
        *,
        template_path: Union[Path, str] = Path(__file__).parent / "vf_template.xml",
        file_to_save: str,
        pci_address: "PCIAddress",
    ) -> Path:
        """
        Create configuration xml of VF for QEMU.

        :param template_path: Path to template config
        :param file_to_save: Path to destination directory
        :param pci_address: PCI Address of VF
        :return: Path to saved configuration
        """
        logger.log(log_levels.MODULE_DEBUG, msg="Creating configuration file for VF.")
        data_for_template = {}
        for key, value in asdict(pci_address).items():
            # convert int to hex required for xml
            data_for_template[key] = hex(value)
        file_to_save = self._render_file(file_to_save, data_for_template, template_path)
        return file_to_save

    def _render_file(
        self,
        file_to_save: str,
        data_for_template: Dict,
        template_path: Union[Path, str],
    ) -> Path:
        """
        Create configuration xml for QEMU.

        :param template_path: Path to template config
        :param file_to_save: Path to destination directory
        :param data_for_template: Parameters dict to pass into template
        :return: Path to saved configuration
        """
        if isinstance(template_path, Path):
            template_path = str(template_path)

        with open(template_path, "rb") as f:
            template = Environment(autoescape=select_autoescape()).from_string(f.read().decode("UTF-8"))
        logger.log(log_levels.MODULE_DEBUG, msg="Opening file for generating configuration.")
        file_to_save = self._conn.path(file_to_save)
        template_config = str(template.render(**data_for_template))
        logger.log(log_levels.MODULE_DEBUG, msg=f"Writing text to file:\n{template_config}...")
        file_to_save.write_text(template_config)

        return file_to_save

    def detach_vf(self, *, name: str, vf_config: str) -> None:
        """
        Detach from VM via config xml of VF.

        :param name: Name of VM
        :param vf_config: Config file with VF
        """
        self.detach_device(name=name, device_config=vf_config)

    def detach_device(self, *, name: str, device_config: str) -> None:
        """
        Detach device from VM via config xml of device.

        :param name: Name of VM
        :param device_config: Config file of the device
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.detach_device(
            name=name,
            device_config=device_config,
            status=self.get_vm_status(name)["State"],
        )

    def attach_vf(self, *, name: str, vf_config: str) -> None:
        """
        Attach VF to VM.

        :param name: Name of VM
        :param vf_config: Config file with VF
        """
        self.attach_device(name=name, device_config=vf_config)

    def attach_agent(self, *, name: str, agent_config_file: str) -> None:
        """
        Attach agent to VM.

        :param name: Name of VM
        :param agent_config_file: Agent config file
        """
        self.attach_device(name=name, device_config=agent_config_file)

    def attach_device(self, name: str, device_config: str) -> None:
        """
        Attach device.

        :param name: Name of VM
        :param device_config: Config file with device
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.attach_device(
            name=name,
            device_config=device_config,
            status=self.get_vm_status(name)["State"],
        )

    def clone_vm_hdd_image(
        self,
        *,
        path_to_source_image: Path,
        path_to_destination_image: Path,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Path:
        """
        Clone hdd image for vm from source into given name.

        :param path_to_source_image: Path to source image to clone
        :param path_to_destination_image: Path to destination image
        :param timeout: Time to clone image
        :raises KVMHypervisorException: if cloning failed or timeout occurred
        :raises FileNotFoundError: if not found path_to_source_image
        :return: Path for cloned or already matching hdd image
        """
        path_to_source_image = self._conn.path(str(path_to_source_image))
        path_to_destination_image = self._conn.path(str(path_to_destination_image))
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Cloning disk image {path_to_source_image} into {path_to_destination_image}, "
            f"timeout - {timeout} secs.",
        )
        if path_to_source_image.is_file():
            if path_to_destination_image.exists():
                source_checksum_path = path_to_source_image.parent / f"{path_to_source_image.name}.md5"
                if source_checksum_path.is_file():
                    checksum_file_content = source_checksum_path.read_text().strip()
                    source_checksum = checksum_file_content.split(maxsplit=1)[0] if checksum_file_content else ""
                else:
                    source_checksum = self._conn.execute_command(f"md5sum {path_to_source_image}").stdout.split()[0]
                destination_checksum = self._conn.execute_command(f"md5sum {path_to_destination_image}").stdout.split()[
                    0
                ]
                if source_checksum.lower() == destination_checksum.lower():
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Destination image {path_to_destination_image} already matches the source; "
                        "skipping cloning.",
                    )
                    return path_to_destination_image
                logger.log(
                    log_levels.MODULE_DEBUG,
                    msg=f"Destination image {path_to_destination_image} differs from the source; removing it.",
                )
                path_to_destination_image.unlink()

            timeout_counter = TimeoutCounter(timeout)
            copy_command = "rsync -aq" if self._rsync_available else "scp"
            # show 5 column of ls, size of file
            target_size = self._conn.execute_command(
                f"ls {path_to_source_image} -l | awk '{{print $5}}'", shell=True
            ).stdout
            copy_process = self._conn.start_process(
                f"{copy_command} {path_to_source_image} {path_to_destination_image}"
            )
            logger.log(log_levels.MODULE_DEBUG, msg="Checking if cloning is completed..")
            while not timeout_counter:
                if not copy_process.running:
                    logger.log(log_levels.MODULE_DEBUG, msg="Cloning completed.")
                    return path_to_destination_image
                current_size = (
                    self._conn.execute_command(
                        f"ls {path_to_destination_image} -l | awk '{{print $5}}'",
                        shell=True,
                    ).stdout
                    if path_to_destination_image.exists()
                    else 0
                )
                try:
                    current_size = int(current_size)
                except (TypeError, ValueError):
                    current_size = 0
                if current_size > 0:
                    logging.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"still cloning... {current_size / int(target_size) * 100:.0f} %, next check in 30secs.",
                    )
                sleep(30)
            else:
                raise KVMHypervisorException(
                    f"Cloning image {path_to_source_image} not finished in given timeout: {timeout}"
                )
        else:
            raise FileNotFoundError(f"Not found {path_to_source_image} in system.")

    def create_mdev(
        self,
        *,
        mdev_uuid: Union[str, "UUID"],
        pci_address: "PCIAddress",
        template_path: Union[Path, str] = Path(__file__).parent / "mdev_template.xml",
        file_to_save: str,
    ) -> Union[str, Path]:
        """
        Create a mediated device by using the PCI address.

        :param mdev_uuid: ID of device
        :param pci_address: Address of PCI
        :param template_path: Path to template config of mdev
        :param file_to_save: Path to destination directory for xml
        :raises KVMHypervisorExecutionException: if creating of device failed
        :raises KVMHypervisorException: if creating mdev command succeed but uuid was not in output
        """
        pci_address_string = pci_address.lspci.replace(":", r"\:")
        command = (
            f'echo "{mdev_uuid}" | tee /sys/class/mdev_bus/{pci_address_string}'
            f"/mdev_supported_types/ice-vdcm/create"
        )
        result = self._conn.execute_command(command, shell=True, custom_exception=KVMHypervisorExecutionException)

        if str(mdev_uuid) not in result.stdout:
            raise KVMHypervisorException(f"{mdev_uuid} not found in cmd output: {result.stdout}")

        return self._render_file(file_to_save, {"uuid": str(mdev_uuid)}, template_path)

    def destroy_mdev(self, mdev_uuid: Union[str, "UUID"]) -> None:
        """
        Destroy the mediated device.

        :param mdev_uuid: ID of device
        :raises KVMHypervisorExecutionException: if destroying device failed
        """
        command = f"echo 1 > /sys/bus/mdev/devices/{mdev_uuid}/remove"
        self._conn.execute_command(command, shell=True, custom_exception=KVMHypervisorExecutionException)

    def get_hdd_path(self, name: str) -> Path:
        """
        Get path of used disk image.

        :param name: Name of VM
        :raises KVMHypervisorExecutionException: if command failed
        :raises KVMHypervisorException: if hdd path not found
        :return: Path of used disk image
        """
        logger.log(log_levels.MODULE_DEBUG, msg="Getting HDD path of used disk image.")

        xml = self.dump_xml(name)
        entry = xml.find(".//source[@file]")
        try:
            hdd_path = entry.attrib["file"]
        except (AttributeError, KeyError):
            raise KVMHypervisorException(f"HDD path for {name} not found in dumped xml!")
        return self._conn.path(hdd_path)

    def dump_xml(self, name: str) -> ET.ElementTree:
        """
        Dump xml.

        :param name: Name of VM
        :return: Parsed output from dumpxml command to xml.etree.ElementTree
        :raises: VirshException if virt_tool fails.
        """
        return self.virt_tool.dump_xml(name)

    @staticmethod
    def _find_pci(node: ET.Element, xpath: str) -> PCIAddress:
        pci = node.find(xpath)
        try:
            pci_domain = int(pci.attrib["domain"], base=16)
            bus = int(pci.attrib["bus"], base=16)
            slot = int(pci.attrib["slot"], base=16)
            func = int(pci.attrib["function"], base=16)
        except (AttributeError, KeyError):
            raise KVMHypervisorException("PCI not found in xml!")
        return PCIAddress(pci_domain, bus, slot, func)

    def get_pci_for_host_vf_and_vm_vf(self, name: str) -> List[Tuple[PCIAddress, PCIAddress]]:
        """
        Get PCIs for Host VF and correlated VM VF.

        :param name: Name of VM
        :raises KVMHypervisorExecutionException: if command failed
        :raises KVMHypervisorException: if Interface with Host VF and VM VF was not found in xml
        :raises KVMHypervisorException: if PCI was not found in xml
        :return: List of tuples with Host VF PCI and VM VF PCI, 1st element in tuple is Host VF PCI and 2nd is VM VF PCI
        """
        logger.log(log_levels.MODULE_DEBUG, msg="Get PCI for Host VF and VM VF.")
        result = []
        xml = self.dump_xml(name)
        entries = xml.findall(".//hostdev[@mode='subsystem']")
        if not entries:
            raise KVMHypervisorException("Interface with Host VF and VM VF not found in xml!")
        for entry in entries:
            host_vf_pci = self._find_pci(entry, ".//source/address")
            vm_vf_pci = self._find_pci(entry, "./address")
            logger.log(
                log_levels.MODULE_DEBUG,
                msg=f"VM: {name}, Host VF PCI: {host_vf_pci}, VM VF PCI: {vm_vf_pci}",
            )
            result.append((host_vf_pci, vm_vf_pci))
        return result

    def get_pci_addresses_of_vfs(self, *, interface: str) -> List[PCIAddress]:
        """
        Get pci address of all VFs created on host using /sys/class/net/interface_name.

        :param interface: PF name that created the VFs.
        :raises VFExceptionKVM: If not found pci address.
        :raises KVMHypervisorExecutionException: if command failed.
        :return: PCI Addresses of VFs.
        """
        pci_addresses = []
        result = self._conn.execute_command(
            f"ls /sys/class/net/{interface}/device/virtfn* -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        matched = re.findall(self.VF_PCI_ADDRESS_REGEX, result.stdout, re.M)
        if matched:
            for pci_address in matched:
                pci_addresses.append(PCIAddress(*[int(data, 16) for data in re.split(r"\W+", pci_address)]))
            return pci_addresses
        else:
            raise VFExceptionKVM(f"Not matched pci addresses for interface {interface}.")

    def get_pci_addresses_of_vfs_by_pci(self, pci_address: PCIAddress) -> List[PCIAddress]:
        """
        Get pci address of all VFs created on host using /sys/bus/pci/devices/pci_address.

        :param pci_address: PCI address of the PF adapter.
        :raises VFExceptionKVM: If not found pci address.
        :raises KVMHypervisorExecutionException: if command failed.
        :return: List PCI Addresses of VFs.
        """
        pci_addresses = []
        result = self._conn.execute_command(
            f"ls /sys/bus/pci/devices/{pci_address}/virtfn* -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        matched = re.findall(self.VF_PCI_ADDRESS_REGEX, result.stdout, re.M)
        if matched:
            for pci_address in matched:
                pci_addresses.append(PCIAddress(*[int(data, 16) for data in re.split(r"\W+", pci_address)]))
            return pci_addresses
        else:
            raise VFExceptionKVM(f"Not matched pci addresses for interface {pci_address}.")

    def get_vf_id_from_pci(self, *, interface: str, pci: PCIAddress) -> int:
        """
        Get ID of VF with the given PCI address on specific PF PCI address using /sys/class/net/interface_name.

        :param interface: Name of PF interface that created the VFs.
        :param pci: PCI address of VF.
        :raises VFExceptionKVM: If VFs are not found.
        :raises KVMHypervisorExecutionException: If command failed.
        :return: ID of VF with the given PCI address on specific PF (interface).
        """
        result = self._conn.execute_command(
            f"ls /sys/class/net/{interface}/device/virtfn* -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        vf_number_regex = rf"^.*device/virtfn(?P<vf_number>\d+).*->.*{pci}$"
        match = re.search(vf_number_regex, result.stdout, re.M)
        if match:
            return int(match.group("vf_number"))
        else:
            raise VFExceptionKVM(f"Not matched VFs for interface {interface}.")

    def get_vf_id_by_pci(self, pf_pci_address: PCIAddress, vf_pci_address: PCIAddress) -> int:
        """
        Get ID of VF with the given PCI address on specific PF PCI address using /sys/bus/pci/devices/pci_address.

        :param pf_pci_address: PF interface PCI address that created VFs.
        :param vf_pci_address: VF interface PCI address.
        :return: ID of the VF.
        """
        result = self._conn.execute_command(
            f"ls /sys/bus/pci/devices/{pf_pci_address}/virtfn* -la",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        vf_number_regex = rf"^.*devices/{pf_pci_address}/virtfn(?P<vf_number>\d+).*->.*{vf_pci_address}$"
        match = re.search(vf_number_regex, result.stdout, re.M)
        if match:
            return int(match.group("vf_number"))
        else:
            raise VFExceptionKVM(f"Not matched VFs for PF PCI Address {pf_pci_address}")

    def get_vfs_details_from_interface(self, *, interface_name: str) -> List[VFDetail]:
        """
        Get VF details per provided PF interface name.

        :param interface_name: PF interface name
        :raises: KVMHypervisorException in case of command failure (rc != 0)
        :return: List of VFDetail objects
        """
        command = f"ip link show dev {interface_name}"
        pattern = (
            r"vf\s+(?P<vf_id>\d+)\D+(?P<mac>([a-f0-9]{2}[:]){5}[a-f0-9]{2})"
            r".+spoof\schecking\s(?P<spoofchk>\w+).+trust\s+(?P<trust>\w+)"
        )

        output = self._conn.execute_command(command=command, custom_exception=KVMHypervisorExecutionException).stdout
        vf_details = []
        for line in output.splitlines():
            match = re.match(string=line, pattern=pattern)
            if match:
                vf_details.append(
                    VFDetail(
                        id=int(match.group("vf_id")),
                        mac_address=MACAddress(match.group("mac")),
                        spoofchk=match.group("spoofchk") == "on",
                        trust=match.group("trust") == "on",
                    )
                )
        return vf_details

    def get_vf_id_from_mac_address(self, *, interface_name: str, mac_address: MACAddress) -> int:
        """
        Get ID of VF by using provided MAC Address and interface name (PF).

        In case of multiple VFs sharing same MAC Address first item will be returned.
        :param interface_name: PF Interface name
        :param mac_address: MAC Address of VF
        :raises VFExceptionKVM: If not found VF
        :raises KVMHypervisorExecutionException: If command failed
        :return: ID of VF
        """
        vfs_details = self.get_vfs_details_from_interface(interface_name=interface_name)

        for vf_detail in vfs_details:
            if vf_detail.mac_address == mac_address:
                return vf_detail.id

    def create_bridge(self, bridge_name: str) -> None:
        """
        Add new bridge to the system.

        :param bridge_name: Name of bridge to create
        :raises KVMHypervisorExecutionException: If creating bridge failed
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Create bridge: {bridge_name}")
        command = f"brctl addbr {bridge_name}"
        self._conn.execute_command(command, shell=True, custom_exception=KVMHypervisorExecutionException)

    def delete_bridge(self, bridge_name: str) -> None:
        """
        Delete bridge from the system.

        :param bridge_name: Name of bridge to delete
        :raises KVMHypervisorExecutionException: If deleting bridge failed
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Delete bridge: {bridge_name}")
        command = f"brctl delbr {bridge_name}"
        self._conn.execute_command(command, shell=True, custom_exception=KVMHypervisorExecutionException)

    def add_interface_to_bridge(self, *, bridge_name: str, interface: str) -> None:
        """
        Connect interface to bridge.

        :param bridge_name: Name of bridge where interface should be added
        :param interface: Friendly name (eg. ethX) of interface which will be added to bridge
        :raises KVMHypervisorExecutionException: If deleting bridge failed
        """
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Connect interface: {interface} to bridge: {bridge_name}",
        )
        command = f"brctl addif {bridge_name} {interface}"
        self._conn.execute_command(command, shell=True, custom_exception=KVMHypervisorExecutionException)

    def detach_interfaces(self, vm_names_list: List[str]) -> None:
        """
        Detach interfaces from given VMs.

        For each VM, fetch all VF pci addresses and detach them all. In the event of a failure, this function will
        continue to attempt to detach all VFs before returning.

        :param vm_names_list: List of VM names
        """
        for vm_name in vm_names_list:
            try:
                matched_vf_pcis = self.get_pci_for_host_vf_and_vm_vf(vm_name)
                logger.log(log_levels.MODULE_DEBUG, msg=f"Matched VF PCIs: {matched_vf_pcis}")
            except KVMHypervisorException as e:
                if "Interface with Host VF and VM VF not found in xml!" in e.args:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"No VFs present on {vm_name}, nothing to detach.",
                    )
                    continue
            for host_pci, vm_pci in matched_vf_pcis:
                iface_xml = self._conn.path(self._conn.path.cwd(), "iface.xml")
                vf_config = self.prepare_vf_xml(pci_address=host_pci, file_to_save=str(iface_xml))

                logger.log(log_levels.MODULE_DEBUG, msg=f"Detach VF {vm_pci}")
                try:
                    self.detach_vf(name=vm_name, vf_config=vf_config)
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Interface {vm_pci} detached from {vm_name}",
                    )
                except KVMHypervisorExecutionException as e:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Interface {vm_pci} couldn't be detached from {vm_name}: {e}",
                    )

    def get_dynamic_ram(
        self,
        *,
        vm_number: int,
        vm_min_ram: int = 2000,
        vm_max_ram: Optional[int] = 10000,
        reserved_memory: Optional[int] = 10000,
    ) -> int:
        """
        Get calculated RAM per VM based on available memory and vm number.

        :param vm_number: Number of planned VMs to create
        :param vm_min_ram: Minimal RAM for VM in MB
        :param vm_max_ram: Maximal RAM for VM in MB
        :param reserved_memory: Reserved RAM for host to operate in MB
        :return: RAM per VM in MB, if no output from awk command then vm_min_ram is returned
        :raises KVMHypervisorException: If not enough free RAM for on SUT for VM
        """
        free_awk = (
            "free -m | awk '{ if (NR==1) { if (/cached$/) {version=0} else"
            "{version=1} } else if (/^Mem/) { if (version==0)"
            "{print $4 + $6 + $7} else {print $7} } }'"
        )
        result = self._conn.execute_command(free_awk, shell=True)
        if result.return_code == 0:
            free_ram = int(result.stdout)
            if vm_min_ram * vm_number >= free_ram:
                raise KVMHypervisorException("Not enough free RAM on SUT for VM.")

            if free_ram >= reserved_memory + vm_min_ram * vm_number:
                mem = (free_ram - reserved_memory) / vm_number
                if mem > vm_max_ram:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Reducing RAM per VM to {vm_max_ram} max",
                    )
                    mem = vm_max_ram
            else:
                mem = vm_min_ram

            logger.log(
                log_levels.MODULE_DEBUG,
                msg=f"Dynamic RAM setting option calculated {mem} MB RAM per VM",
            )
            return mem

        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"There's not output from awk, proceeding with default {vm_min_ram} MB",
        )
        return vm_min_ram

    def get_mdev_details(self, name: str) -> List[Tuple[str, PCIAddress]]:
        """
        Get PCIs and UUIDs of MDEV for VM.

        :param name: Name of VM
        :raises KVMHypervisorExecutionException: if command failed
        :raises KVMHypervisorException: if Interface with mdev was not found in xml
        :raises KVMHypervisorException: if PCI was not found in xml
        :raises KVMHypervisorException: if UUID was not found in xml
        :return: List of tuples with Host MDEV and VM VF PCI, 1st element in tuple is MDEV and 2nd is VM VF PCI
        """
        logger.log(log_levels.MODULE_DEBUG, msg="Get PCI and UUID for MDEV.")

        result = []
        xml = self.dump_xml(name)
        entries = xml.findall(".//hostdev[@mode='subsystem'][@type='mdev']")
        if not entries:
            raise KVMHypervisorException("Interface with MDEV not found in xml!")
        for entry in entries:
            pci = entry.find(".//source/address")
            host_uuid = pci.attrib.get("uuid")
            if host_uuid is None:
                raise KVMHypervisorException("Interface with MDEV does not contains UUID!")
            vm_vf_pci = self._find_pci(entry, "./address")
            logger.log(
                log_levels.MODULE_DEBUG,
                msg=f"VM: {name}, Host UUID: {host_uuid}, VM VF PCI: {vm_vf_pci}",
            )
            result.append((host_uuid, vm_vf_pci))
        return result

    def get_pci_address_of_mdev_pf(self, mdev_uuid: Union[str, "UUID"]) -> PCIAddress:
        """
        Get PCI address of mediated device PF.

        :param mdev_uuid: ID of device
        :raises KVMHypervisorExecutionException: if command failed
        :raises VFExceptionKVM: If PCI address not found
        :return: PCIAddress of mediated device PF
        """
        result = self._conn.execute_command(
            f"cat /sys/bus/mdev/devices/{mdev_uuid}/mdev_type/name",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        match = re.search(self.VF_PCI_ADDRESS_REGEX, result.stdout)
        if match:
            pci_address_data = [int(data, 16) for data in re.split(r"\W+", match.group("address"))]
            return PCIAddress(*pci_address_data)
        raise VFExceptionKVM(f"Not matched PF PCI for MDEV with UUID: {str(mdev_uuid)}")

    def get_all_mdev_uuids(self) -> List[str]:
        """
        Get all MDEV UUIDs.

        :raises KVMHypervisorExecutionException: if command failed
        :raises KVMHypervisorException: if MDEV UUIDs not found
        :return: List of UUIDs
        """
        logger.log(log_levels.MODULE_DEBUG, msg="Get all MDEV UUIDs.")

        result = self._conn.execute_command(
            command="ls /sys/bus/mdev/devices",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        match = re.findall(self.UUID_REGEX, result.stdout, re.M)
        if match:
            return match
        raise KVMHypervisorException(f"MDEV UUIDs not found!: {result.stdout}")

    def set_trunk(self, pf_interface: str, action: str, vlan_id: str, vf_id: int) -> None:
        """
        Set trunk VF-d value, VLAN to filter on. Supports two operations, add and rem.

        :param pf_interface: PF interface name.
        :param action: add or rem.
        :param vlan_id: VLAN ID
        :param vf_id: VF ID.
        :raises KVMHypervisorException: If action is incorrect.
        :raises KVMHypervisorExecutionException: if command failed.
        """
        if action not in ["add", "rem"]:
            raise KVMHypervisorException(f"Unsupported action: {action}, please use 'add' or 'rem'.")

        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Configure trunk VLAN {vlan_id} for a VF {vf_id}.",
        )
        self._conn.execute_command(
            f"echo {action} {vlan_id} > /sys/class/net/{pf_interface}/device/sriov/{vf_id}/trunk",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )

    def get_trunk(self, pf_interface: str, vf_id: int) -> str:
        """
        Get trunk VF-d value.

        :param pf_interface: PF interface name.
        :param vf_id: VF ID.
        :raises KVMHypervisorExecutionException: if command failed.
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Get current trunk value for a VF {vf_id}.")
        result = self._conn.execute_command(
            f"cat /sys/class/net/{pf_interface}/device/sriov/{vf_id}/trunk",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        return result.stdout.rstrip()

    def set_number_of_vfs_for_pf_by_pci(
        self,
        pci_address: "PCIAddress",
        vfs_count: int,
        check: bool = True,
        timeout: int = 60,
    ) -> None:
        """
        Set VFs by /sys/bus/pci/devices/pci_address/sriov_numvfs.

        :param pci_address: PF VF PCI Address.
        :param vfs_count: What to set sriov_numvfs/how many VFs to set.
        :param check: When True check if created number of VFs matches required number of VFs.
        :param timeout: Max time to execute (secs)
        :raises KVMHypervisorExecutionException: If command failed.
        """
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Configuring {vfs_count} VFs for PCI Address {pci_address}.",
        )
        result = self._conn.execute_command(
            f"echo {vfs_count} > /sys/bus/pci/devices/{pci_address}/sriov_numvfs",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
            expected_return_codes={0, 1},
            timeout=timeout,
        )
        busy_message = "echo: write error: Device or resource busy"
        # if VFs are defined, need to disable sriov
        if result.return_code == 1 and busy_message in result.stderr:
            logger.log(
                log_levels.MODULE_DEBUG,
                msg="VFs are already configured, disabling and trying again.",
            )
            self._conn.execute_command(
                f"echo 0 > /sys/bus/pci/devices/{pci_address}/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=timeout,
            )
            self._conn.execute_command(
                f"echo {vfs_count} > /sys/bus/pci/devices/{pci_address}/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=timeout,
            )
        if check:
            self.check_number_of_vfs_by_pci(pci_address=pci_address, vfs_count=vfs_count)

    def check_number_of_vfs_by_pci(self, pci_address: "PCIAddress", vfs_count: int) -> None:
        """
        Check if number of VFs is correct using /sys/bus/pci/devices/pci_address.

        :param pci_address: PF VF PCI Address.
        :param vfs_count: Expected number of VFs.
        :raises VFExceptionKVM: If count of VFs is different from expected.
        :raises KVMHypervisorExecutionException: If command failed.
        :raises TimeoutExpired: If timeout occurred.
        """
        result = self._conn.execute_command(
            f"ls /sys/bus/pci/devices/{pci_address}/virtfn* -la",
            shell=True,
            expected_return_codes={0, 2},
            custom_exception=KVMHypervisorExecutionException,
        )
        if "No such file or directory" in result.stderr:  # no VFs were created
            created_vfs_count = 0
        else:
            vf_regex = rf"^.*{pci_address}/virtfn(?P<vf_number>\d+).*$"
            match = [m.groupdict() for m in re.finditer(vf_regex, result.stdout, re.M)]
            created_vfs_count = len(match)

        if created_vfs_count != vfs_count:
            raise VFExceptionKVM(
                f"Mismatched count of created and expected VFs. Created: {created_vfs_count}, expected: {vfs_count}"
            )
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Correctly assigned {vfs_count} for PF with PCI {pci_address}.",
        )

    def set_tpid(self, interface: str, tpid: str) -> None:
        """
        Set TPID value, Specifies the TPID of the outer VLAN tag (S-tag).

        :param interface: Interface name.
        :param tpid: TPID value, eg: 88a8.
        :raises KVMHypervisorExecutionException: if command failed.
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Configure TPID {tpid} for {interface}.")
        self._conn.execute_command(
            f"echo {tpid} > /sys/class/net/{interface}/device/sriov/tpid",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )

    def get_tpid(self, interface: str) -> str:
        """
        Get TPID value.

        :param interface: Interface name.
        :raises KVMHypervisorExecutionException: if command failed.
        """
        logger.log(log_levels.MODULE_DEBUG, msg=f"Get current TPID value for {interface}.")
        result = self._conn.execute_command(
            f"cat /sys/class/net/{interface}/device/sriov/tpid",
            shell=True,
            custom_exception=KVMHypervisorExecutionException,
        )
        return result.stdout.rstrip()

    def prepare_pci_controller_xml(
        self,
        *,
        template_path: Union[Path, str] = Path(__file__).parent / "pci_controller_template.xml",
        file_to_save: str,
        index: int,
        chassis: hex,
        port: hex,
        pci_address: "PCIAddress",
    ) -> Path:
        """
        Create configuration xml of PCI Controller for q35.

        :param template_path: Path to template config
        :param file_to_save: Path to destination directory
        :param index: Index of PCi Controller
        :param chassis: Chassis of PCi Controller
        :param port: Port for attaching PCi Controller
        :param pci_address: PCI Address of PCI Controller
        :return: Path to saved configuration
        """
        logger.log(
            log_levels.MODULE_DEBUG,
            msg="Creating configuration file for PCI Controller.",
        )
        data_for_template = {}
        for key, value in asdict(pci_address).items():
            # convert int to hex required for xml
            data_for_template[key] = hex(value)
        data_for_template["index"] = index
        data_for_template["chassis"] = chassis
        data_for_template["port"] = hex(port)
        file_to_save = self._render_file(file_to_save, data_for_template, template_path)
        return file_to_save

    def attach_pci_controllers(
        self,
        name: str,
        number_of_devices: int,
        first_index: int,
        first_chassis: int,
        first_port: hex,
        first_bus: hex,
        first_slot: hex,
        first_func: hex,
    ) -> None:
        """
        Stop VM, attach PCI controllers and start VM.

        The following parameters should be incrementing with each attach device: index, chassis, port, function.
        The slot (value as hex) should be incrementing when function is grater than 7.
        The function should be between 1-7.
        The port must be <= 0xFF.
        The slot must be <=0x1F.

        :param name: Name of virtual machine.
        :param number_of_devices: Amount of attached PCI Controllers.
        :param first_bus: BUS of first attached PCI Controller.
        :param first_index: Index of first attached PCI Controller.
        :param first_chassis: Chassis of first attached PCI Controller.
        :param first_port: Port of first attached PCI Controller.
        :param first_slot: SLOT of first attached PCI Controller.
        :param first_func: Function of first attached PCI Controller.
        """
        number_of_attached_devs = 0
        bus = first_bus
        func = first_func
        slot = first_slot
        port = first_port
        index = first_index
        chassis = first_chassis
        self.shutdown_vm(name=name)
        while port < 256 and func < 8 and slot < 32:
            pci_device = PCIAddress(bus=bus, domain=0, func=func, slot=slot)
            logger.log(
                log_levels.MODULE_DEBUG,
                f"Attaching PCI Controller for PCI Device: {pci_device}...",
            )
            iface_xml = self._conn.path(self._conn.path.cwd(), "iface.xml")
            device_config = self.prepare_pci_controller_xml(
                file_to_save=str(iface_xml),
                index=index,
                chassis=chassis,
                port=port,
                pci_address=pci_device,
            )
            try:
                self.attach_device(name=name, device_config=str(device_config))
                number_of_attached_devs += 1
            except KVMHypervisorExecutionException as e:
                if "Attempted double use of PCI Address" in e.stderr or "already exists" in e.stderr:
                    logger.log(
                        log_levels.MODULE_DEBUG,
                        msg=f"Failed to attach device, try next PCI address: {e}",
                    )
                else:
                    raise e
            func += 1
            port += 1
            index += 1
            chassis += 1
            if func >= 8:
                func = 1
                slot += 1
            if number_of_attached_devs == number_of_devices:
                break
        if number_of_attached_devs != number_of_devices:
            raise KVMHypervisorException(
                f"Not enough free PCI devices. Cannot create expected number of PCI Controllers: expected: "
                f"{number_of_devices}, created: {number_of_attached_devs}"
            )
        logger.log(
            log_levels.MODULE_DEBUG,
            f"Attached successfully {number_of_attached_devs} PCI Controllers.",
        )

        self.start_vm(name)

    def dump_xml_from_vm(self, host_name: str) -> Union[str, None]:
        """
        Output a guest's XML configuration file.

        :param host_name: Name of the host to get xml from.
        :return: string with xml content or None if fails.
        """
        return self.virt_tool.dump_xml_from_vm(host_name)

    def detach_interface_from_vm(self, guest_name: str, mac: str) -> bool:
        """
        Detach a network interface from a guest.

        :param guest_name: guest to detach interface from.
        :param mac: mac of the interface.
        :return: False if fails
        """
        return self.virt_tool.detach_interface_from_vm(guest_name, mac)

    def list_vms(self, all_vms: bool = True) -> List[OrderedDict[str, str]]:
        """
        List guests.

        :param all_vms: If true lists all VMs, even shutdown ones.
        :return: List of vm guests.
        """
        return self.virt_tool.list_vms(all_vms)

    def get_mac_for_mng_vm_interface(self, guest_name: str) -> str:
        """
        Return MAC address of management interface for Virtual Machine.

        :param guest_name: Name of the virtual machine.
        :return: MAC address on success, None otherwise.
        :raises: VirshException if virt_tool fails or no mac is found.
        """
        return self.virt_tool.get_mac_for_mng_vm_interface(guest_name)

    def get_mng_ip_for_vm(self, mac: str, vm_id: str, tries: int = 300) -> str:
        """
        Get mng IP address for Virtual Machine.

        :param mac: MAC address of Virtual Machine management adapter.
        :param vm_id: ID of the VM.
        :param tries: tries to get IP address from VM.
        :return: mng IP for virtual machine.
        :raises: VMNotRunKVM or VMMngIpNotAvailableKVM if virt_tool fails.
        """
        return self.virt_tool.get_mng_ip_for_vm(mac, vm_id, tries)

    def get_guest_mng_ip(self, vm_id: str, timeout: int = 300) -> str:
        """Get mng IP address for Virtual Machine.

        :param vm_id: ID of VM
        :param timeout: Time to get ip address from VM.
        :return: mng IP for virtual machine or raises KVMHypervisorException.
        """
        try:
            mac = self.get_mac_for_mng_vm_interface(vm_id)
            logger.info(msg=f"Finding mng IP for VM with MAC: {mac}")
        except KVMHypervisorException:
            raise KVMHypervisorException(f"Cannot find MAC address for VM: {vm_id}")

        return self.get_mng_ip_for_vm(mac=mac, vm_id=vm_id, tries=timeout)

    def set_vcpus(self, vm_name: str, nr: int) -> None:
        """
        Change number of vCPUs.

        :param vm_name: name of VM
        :param nr: number of vCPUs
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.set_vcpus(vm_name, nr)

    def set_vcpus_max_limit(self, vm_name: str, nr: int) -> None:
        """
        Change maximum number of vCPUs.

        :param vm_name: name of VM
        :param nr: number of vCPUs
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.set_vcpus_max_limit(vm_name, nr)

    def create_vm_network(self, xml_file: str) -> bool:
        """
        Create network for VM.

        :param xml_file: Name of template file.
        :return: True on success False on fail.
        """
        return self.virt_tool.create_vm_network(xml_file)

    def destroy_vm_network(self, net: str) -> bool:
        """
        Destroy VM network.

        :param net: Network name to destroy.
        :return: True on success False on fail.
        """
        return self.virt_tool.destroy_vm_network(net)

    def attach_tap_interface_to_vm(
        self,
        vm_name: str,
        net: str,
        config: str = "live",
        interface_type: str = "network",
    ) -> bool:
        """
        Attach tap interface to VM.

        :param vm_name: Name of the VM.
        :param net: Network name to attach tap to.
        :param config: type of attachment - live for
        :param interface_type: Interface type eg. network / bridge.
        :return: True on success False on fail
        """
        return self.virt_tool.attach_tap_interface_to_vm(vm_name, net, config, interface_type)

    def attach_interface(self, guest_name: str, pci_address: PCIAddress) -> None:
        """
        Create XML and attach interface (passthrough) to given VM.

        :param guest_name: name of VM
        :param pci_address: PCIAddress of the interface.
        """
        # create adapter xml
        iface_xml = self._conn.path(self._conn.path.cwd(), "iface.xml")
        interface_config = self.prepare_vf_xml(pci_address=pci_address, file_to_save=iface_xml)
        logger.log(log_levels.MODULE_DEBUG, msg=f"Attach VF {pci_address}")
        self.attach_device(name=guest_name, device_config=interface_config)
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Interface {pci_address} attached to {guest_name}",
        )

    def detach_interface(self, guest_name: str, pci_address: PCIAddress) -> None:
        """
        Create XML and detach interface (passthrough) from VM.

        :param guest_name: name of VM
        :param pci_address: PCIAddress of the attached interface.
        """
        # create adapter xml
        iface_xml = self._conn.path(self._conn.path.cwd(), "iface.xml")
        adapter_config = self.prepare_vf_xml(pci_address=pci_address, file_to_save=iface_xml)
        logger.log(log_levels.MODULE_DEBUG, msg=f"Detach interface {pci_address}")
        self.detach_vf(name=guest_name, vf_config=adapter_config)
        logger.log(
            log_levels.MODULE_DEBUG,
            msg=f"Interface {pci_address} detached from {guest_name}",
        )

    def create_vm_from_xml(self, params: "VMParams") -> str:
        """
        Create VM using predefined xml file.

        :param params: Parameters for VM creating
        :raises KVMHypervisorExecutionException: if command failed
        :raises FileNotFoundError: if not found disk path
        :raises KVMHypervisorException: if disk cloning failed or timeout occurred
        :return: Name of started VM
        """
        logger.info(f"Creating VM using virsh define {params.vm_xml_file}")
        target_file = f"/tmp/{params.name}.xml"
        copy(
            src_conn=self._conn,
            dst_conn=self._conn,
            source=params.vm_xml_file,
            target=target_file,
        )
        path_to_source_image = self._conn.path(params.disk)
        cloned_disk_path = None
        if params.clone_disk:
            clone_timeout = params.clone_timeout if params.clone_timeout else self.DEFAULT_TIMEOUT

            path_to_destination_image = (
                self._conn.path(
                    params.target_hd_clone_disk if params.target_hd_clone_disk else path_to_source_image.parent
                )
                / params.name
            )

            cloned_disk_path = self.clone_vm_hdd_image(
                path_to_source_image=path_to_source_image,
                path_to_destination_image=path_to_destination_image,
                timeout=clone_timeout,
            )
        else:
            if self._conn.path(path_to_source_image).is_file():
                cloned_disk_path = path_to_source_image
                logger.log(
                    log_levels.MODULE_DEBUG,
                    msg=f"Using {path_to_source_image} image for new VM. - 'clone_disk' parameter set to False.",
                )
            else:
                logger.log(
                    log_levels.MODULE_DEBUG,
                    msg=f"NOT using {path_to_source_image} image for new VM. file does not exists.",
                )

        # change disk name in xml file if defined
        if cloned_disk_path:
            command = f"sed -i 's|<VM_DISK>|{cloned_disk_path}|g' {target_file}"
            self._conn.execute_command(command)
        # change MAC and VM-NAME in xml file
        command = f"sed -i 's/<VM_MNG_MAC>/{params.mac_address}/g' {target_file}"
        self._conn.execute_command(command)
        command = f"sed -i 's/<VM_NAME>/{params.name}/g' {target_file}"
        self._conn.execute_command(command)
        # change VM uuid
        vm_uuid = uuid4()
        command = f"sed -i 's/<VM_UUID>/{vm_uuid}/g' {target_file}"
        self._conn.execute_command(command)

        res = self._conn.execute_command(command=f"cat {target_file}", expected_return_codes={0})
        logger.log(log_levels.MODULE_DEBUG, msg=f"VM definition: {res.stdout}")

        self.define_vm(target_file)
        self.start_vm(params.name)
        return params.name

    def define_vm(self, xml_file: str) -> None:
        """
        Define VM.

        :param xml_file: Config file with VM
        :raises: VirshException if virt_tool fails.
        """
        self.virt_tool.define(xml_file)
