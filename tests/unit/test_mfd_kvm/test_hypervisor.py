# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: MIT
"""Tests for `mfd_kvm` package."""

import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path, PosixPath, WindowsPath
from sys import platform
from textwrap import dedent

import pytest
from mfd_common_libs import log_levels
from mfd_connect import RPyCConnection, LocalConnection
from mfd_connect.base import ConnectionCompletedProcess
from mfd_connect.process.rpyc import RPyCProcess
from mfd_typing import MACAddress, PCIAddress, OSName
from netaddr import IPAddress

from mfd_kvm import KVMHypervisor, VMParams, VirshInterface
from mfd_kvm.data_structures import VFDetail
from mfd_kvm.exceptions import (
    KVMHypervisorException,
    KVMHypervisorExecutionException,
    VFExceptionKVM,
    NotFoundInterfaceKVM,
    VirshException,
)


class TestKVMHypervisor:
    @pytest.fixture()
    def hv(self, mocker):
        mocker.patch("mfd_kvm.VirshInterface.__init__", return_value=None)
        conn = mocker.create_autospec(RPyCConnection)
        conn.get_os_name.return_value = OSName.LINUX
        hv = KVMHypervisor(connection=conn)
        hv.virt_tool = mocker.create_autospec(VirshInterface)
        return hv

    def test_get_name_from_ip(self):
        assert KVMHypervisor.get_name_from_ip(IPAddress("10.10.10.1"), "prefix") == "prefix-010-001"

    def test_get_free_network_data(self, hv, mocker):
        mocker.patch("mfd_kvm.hypervisor.sleep")
        hv.parse_network_data_conf = mocker.create_autospec(
            hv.parse_network_data_conf,
            return_value=[
                (IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62")),
                (IPAddress("10.10.10.11"), MACAddress("AA:BB:CC:DD:EE:63")),
            ],
        )
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", return_code=0),
            ConnectionCompletedProcess(args="", return_code=2),
        ]
        assert hv.get_free_network_data(config_file=None, count=1) is not None
        assert hv._conn.execute_command.call_count == 2

    def test_get_free_network_data_multiple(self, hv, mocker):
        mocker.patch("mfd_kvm.hypervisor.sleep")
        hv.parse_network_data_conf = mocker.create_autospec(
            hv.parse_network_data_conf,
            return_value=[
                (IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62")),
                (IPAddress("10.10.10.11"), MACAddress("AA:BB:CC:DD:EE:63")),
            ],
        )
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", return_code=2),
            ConnectionCompletedProcess(args="", return_code=2),
        ]
        assert sorted(hv.get_free_network_data(config_file=None, count=2)) == [
            (IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62")),
            (IPAddress("10.10.10.11"), MACAddress("AA:BB:CC:DD:EE:63")),
        ]
        assert hv._conn.execute_command.call_count == 2

    def test_get_free_network_data_empty_list(self, hv, mocker):
        mocker.patch("mfd_kvm.hypervisor.sleep")
        hv.parse_network_data_conf = mocker.create_autospec(hv.parse_network_data_conf, return_value=[])
        with pytest.raises(KVMHypervisorException):
            hv.get_free_network_data(config_file=None, count=1)

    def test_get_free_network_data_not_found(self, hv, mocker):
        mocker.patch("mfd_kvm.hypervisor.sleep")
        hv.parse_network_data_conf = mocker.create_autospec(
            hv.parse_network_data_conf,
            return_value=[(IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62"))],
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        with pytest.raises(KVMHypervisorException):
            hv.get_free_network_data(config_file=None, count=1)

    def test_parse_network_data_conf_empty_config(self, hv, mocker):
        mocker.patch("mfd_kvm.hypervisor.open", mocker.mock_open(read_data=""))
        with pytest.raises(KVMHypervisorException):
            hv.parse_network_data_conf("")

    def test_parse_network_data_conf(self, hv, mocker):
        data = dedent(
            """\
        [kvm]
        10.10.10.10 AA:BB:CC:DD:EE:62
        10.10.10.11 AA:BB:CC:DD:EE:63"""
        )
        mocker.patch("mfd_kvm.hypervisor.open", mocker.mock_open(read_data=data))
        assert hv.parse_network_data_conf("") == [
            (IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62")),
            (IPAddress("10.10.10.11"), MACAddress("AA:BB:CC:DD:EE:63")),
        ]

    def test_get_list_of_vms_empty_list(self, hv):
        hv.virt_tool.list_vms.return_value = []
        assert hv.get_list_of_vms() == []
        hv.virt_tool.list_vms.assert_called_once()

    def test_get_list_of_vms(self, hv):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        assert hv.get_list_of_vms() == [
            "foo-055-045",
            "Base_R82.img_VM001_10.10.10.11",
        ]
        hv.virt_tool.list_vms.assert_called_once()

    def test_get_vm_status(self, hv):
        expected_status = {
            "Id": "1",
            "Name": "foo-055-045",
            "UUID": "d08c195c-bddb-4fbb-b7aa-0fa10a96c5b1",
            "OS Type": "hvm",
            "State": "running",
            "CPU(s)": "2",
            "CPU time": "109.8s",
            "Max memory": "2097152 KiB",
            "Used memory": "2097152 KiB",
            "Persistent": "yes",
            "Autostart": "disable",
            "Managed save": "no",
            "Security model": "none",
            "Security DOI": "0",
        }
        hv.virt_tool.get_vm_status.return_value = expected_status
        assert hv.get_vm_status("foo-055-045") == expected_status
        hv.virt_tool.get_vm_status.assert_called_once()

    @pytest.mark.parametrize("prefix", ["pre_1", "pre_2"])
    def test_create_multiple_vms(self, hv, mocker, prefix):
        result = [
            (f"{prefix}-010-010", IPAddress("10.10.10.10")),
            (f"{prefix}-010-011", IPAddress("10.10.10.11")),
        ]
        count = 2
        data = [
            (IPAddress("10.10.10.10"), MACAddress("AA:BB:CC:DD:EE:62")),
            (IPAddress("10.10.10.11"), MACAddress("AA:BB:CC:DD:EE:63")),
        ]
        hv.get_free_network_data = mocker.create_autospec(hv.get_free_network_data, return_value=data)
        hv.get_name_from_ip = mocker.create_autospec(
            hv.get_name_from_ip, side_effect=[f"{prefix}-010-010", f"{prefix}-010-011"]
        )
        hv.create_vm = mocker.create_autospec(hv.create_vm, side_effect=[f"{prefix}-010-010", f"{prefix}-010-011"])
        assert (
            hv.create_multiple_vms(
                count=count,
                params=VMParams(),
                ip_data_config_file="test",
                prefix=prefix,
            )
            == result
        )
        hv.get_free_network_data.assert_called_once()
        assert hv.get_name_from_ip.call_count == count
        assert hv.create_vm.call_count == count

    def test_create_vm_error(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(1, "", "", "")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.create_vm(VMParams())

    def test_create_vm_error__osinfo_handle(self, hv, mocker):
        stderr = """
                --os-variant/--osinfo OS name is required, but no value was
               set or detected.
               This is now a fatal error. Specifying an OS name is required
               for modern, performant, and secure virtual machine defaults.
               You can see a full list of possible OS name values with:
                  virt-install --osinfo list
               If your Linux distro is not listed, try one of generic values
               such as: linux2022, linux2020, linux2018, linux2016
               If you just need to get the old behavior back, you can use:
                 --osinfo detect=on,require=off
               Or export VIRTINSTALL_OSINFO_DISABLE_REQUIRE=1"""
        hv._conn.execute_command.side_effect = [
            KVMHypervisorExecutionException(1, "", "", stderr),
            ConnectionCompletedProcess(args="", return_code=0),
        ]
        expected_path = WindowsPath(r"\away\foo.img") if platform == "win32" else PosixPath(r"/away/foo.img")
        expected_command = (
            "virt-install --name=foo.img --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path} --boot=hd,uefi --graphics none --osinfo detect=on,"
            f"require=off"
        )
        source_disk = "/home/disk.img"
        target_disk = Path("/away/")
        vm_config = VMParams(
            name="foo.img",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            target_hd_clone_disk=target_disk,
        )
        source_disk_name = vm_config.name
        destination_disk_path = target_disk / source_disk_name
        hv.clone_vm_hdd_image = mocker.create_autospec(hv.clone_vm_hdd_image, return_value=destination_disk_path)
        hv._conn.path.return_value = destination_disk_path
        assert hv.create_vm(vm_config) == "foo.img"
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)

    def test_create_vm_error_instalation_method_handle(self, hv, mocker):
        stderr = """
                ERROR
                An install method must be specified
                (--location URL, --cdrom CD/ISO, --pxe, --import, --boot hd|cdrom|...)"""
        hv._conn.execute_command.side_effect = [
            KVMHypervisorExecutionException(1, "", "", stderr),
            ConnectionCompletedProcess(args="", return_code=0),
        ]
        expected_path = WindowsPath(r"\away\foo.img") if platform == "win32" else PosixPath(r"/away/foo.img")
        expected_command = (
            "virt-install --name=foo.img --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path} --boot=hd,uefi --graphics none --import"
        )
        source_disk = "/home/disk.img"
        target_disk = Path("/away/")
        vm_config = VMParams(
            name="foo.img",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            target_hd_clone_disk=target_disk,
        )
        source_disk_name = vm_config.name
        destination_disk_path = target_disk / source_disk_name
        hv.clone_vm_hdd_image = mocker.create_autospec(hv.clone_vm_hdd_image, return_value=destination_disk_path)
        hv._conn.path.return_value = destination_disk_path
        assert hv.create_vm(vm_config) == "foo.img"
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)

    def test_create_vm(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=network,hd,uefi --graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_from_xml(self, hv, mocker):
        mocker.patch("mfd_kvm.VirshInterface.__init__", return_value=None)
        conn = mocker.create_autospec(LocalConnection)
        conn.get_os_name.return_value = OSName.LINUX
        hv = KVMHypervisor(connection=conn)
        hv.virt_tool = mocker.create_autospec(VirshInterface)

        hv._conn._ip = "10.10.10.10"
        vm_config = VMParams(
            name="KVM_VM_0",
            vm_xml_file="/var/lib/libvirt/images/ubuntu_kvm_vm_0.xml",
            disk="/var/lib/libvirt/images/ubuntu2004_11072022.qcow2",
            clone_disk=False,
            mac_address="00:11:22:33:44:55",
        )
        target_file = "/tmp/KVM_VM_0.xml"

        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout="<VM xml file>", return_code=0
        )
        expected_uuid = uuid.UUID("d08c195c-bddb-4fbb-b7aa-0fa10a96c5b1")
        with mocker.patch("mfd_kvm.hypervisor.uuid4", return_value=expected_uuid):
            vm_name = hv.create_vm_from_xml(vm_config)

        assert hv.create_vm_from_xml(vm_config) == "KVM_VM_0" == vm_name
        expected_command = f"sed -i 's/<VM_UUID>/{expected_uuid}/g' {target_file}"
        hv._conn.execute_command.assert_any_call(expected_command)

    def test_create_vm_with_arch(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --arch aarch64 --boot=network,hd,uefi --graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            arch="aarch64",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_with_disk_bus(self, hv, mocker):
        expected_path = WindowsPath(r"\away\image.img") if platform == "win32" else PosixPath(r"/away/image.img")
        expected_command = (
            "virt-install --name=image.img --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path},bus=scsi --boot=hd,uefi --graphics none"
        )
        source_disk = "/home/disk.img"
        target_disk = Path("/away/")
        vm_config = VMParams(
            name="image.img",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            disk_bus="scsi",
            target_hd_clone_disk=target_disk,
        )
        source_disk_name = vm_config.name
        destination_disk_path = target_disk / source_disk_name
        hv.clone_vm_hdd_image = mocker.create_autospec(hv.clone_vm_hdd_image, return_value=destination_disk_path)
        hv._conn.path.return_value = destination_disk_path
        assert hv.create_vm(vm_config) == "image.img"
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)

    def test_create_vm_with_disk_bus_arch(self, hv, mocker):
        expected_path = WindowsPath(r"\away\image.img") if platform == "win32" else PosixPath(r"/away/image.img")
        expected_command = (
            "virt-install --name=image.img --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path},bus=scsi --arch aarch64 --boot=hd,uefi --graphics none"
        )
        source_disk = "/home/disk.img"
        target_disk = Path("/away/")
        vm_config = VMParams(
            name="image.img",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            disk_bus="scsi",
            arch="aarch64",
            target_hd_clone_disk=target_disk,
        )
        source_disk_name = vm_config.name
        destination_disk_path = target_disk / source_disk_name
        hv.clone_vm_hdd_image = mocker.create_autospec(hv.clone_vm_hdd_image, return_value=destination_disk_path)
        hv._conn.path.return_value = destination_disk_path
        assert hv.create_vm(vm_config) == "image.img"
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)

    def test_create_vm_with_custom_bridge(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:virbr0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=network,hd,uefi --graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            bridge_name="virbr0",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_with_boot_order(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=hd,network,uefi --graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            boot_order="hd,network",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_with_threads_specified(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2,threads=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=hd,network,uefi --graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            threads=2,
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            boot_order="hd,network",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    @pytest.mark.parametrize("clone_timeout", [None, 6000])
    def test_create_vm_with_disk_path(self, hv, mocker, clone_timeout):
        expected_path = WindowsPath(r"\away\image.img") if platform == "win32" else PosixPath(r"/away/image.img")
        expected_command = (
            "virt-install --name=image.img --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path} --boot=hd,uefi --graphics none"
        )
        source_disk = "/home/disk.img"
        target_disk = Path("/away/")
        vm_config = VMParams(
            name="image.img",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            clone_timeout=clone_timeout,
            target_hd_clone_disk=target_disk,
        )

        source_disk_name = vm_config.name
        destination_disk_path = target_disk / source_disk_name

        hv.clone_vm_hdd_image = mocker.create_autospec(hv.clone_vm_hdd_image, return_value=destination_disk_path)
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "image.img"
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)
        if clone_timeout is not None:
            hv.clone_vm_hdd_image.assert_called_with(
                path_to_source_image=mocker.ANY,
                path_to_destination_image=mocker.ANY,
                timeout=clone_timeout,
            )
        else:
            hv.clone_vm_hdd_image.assert_called_with(
                path_to_source_image=mocker.ANY,
                path_to_destination_image=mocker.ANY,
                timeout=1000,
            )

    def test_create_vm_no_cloning(self, hv):
        source_disk = "/home/disk.img"
        expected_path = WindowsPath(r"\home\disk.img") if platform == "win32" else PosixPath(r"/home/disk.img")
        name = "whatever"
        expected_command = (
            f"virt-install --name={name} --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            f"--os-variant=rhel8.1 --disk path={expected_path} --boot=hd,uefi --graphics none"
        )
        vm_config = VMParams(
            name=name,
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            disk=source_disk,
            clone_disk=False,
        )
        hv._conn.path.return_value = expected_path
        assert hv.create_vm(vm_config) == name
        hv._conn.execute_command.assert_called_with(expected_command, custom_exception=KVMHypervisorExecutionException)

    def test_create_vm_with_graphics(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=network,hd,uefi "
            "--graphics vnc"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            graphics="vnc",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_with_cpu(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=2 --machine=pc --noautoconsole "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=network,hd,uefi "
            "--graphics none --cpu=core2duo,+x2apic,disable=vmx"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            cpu="core2duo,+x2apic,disable=vmx",
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_create_vm_with_large_vcpu(self, hv):
        expected_command = (
            "virt-install --name=foo --memory=1024 --vcpus=256 --machine=pc --noautoconsole "
            "--iommu model=intel,driver.intremap=on,driver.eim=on,driver.caching_mode=on "
            "--features apic=on,ioapic.driver=qemu "
            "--network=bridge:br0,mac=aa:bb:cc:dd:ee:62,model=virtio "
            "--os-variant=rhel8.1 --disk=none --boot=network,hd,uefi "
            "--graphics none"
        )
        vm_config = VMParams(
            name="foo",
            os_variant="rhel8.1",
            mac_address=MACAddress("AA:BB:CC:DD:EE:62"),
            cpu_count=256,
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        assert hv.create_vm(vm_config) == "foo"
        hv._conn.execute_command.assert_called_once_with(
            expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_shutdown_gracefully_vm(self, hv):
        hv.shutdown_gracefully_vm("VM_Name")
        hv.virt_tool.shutdown_gracefully_vm.assert_called_once()

    def test_reboot_vm(self, hv):
        hv.reboot_vm("VM_Name")
        hv.virt_tool.reboot_vm.assert_called_once()

    def test_reset_vm(self, hv):
        hv.reset_vm("VM_Name")
        hv.virt_tool.reset_vm.assert_called_once()

    def test_shutdown_vm(self, hv):
        hv.shutdown_vm("VM_Name")
        hv.virt_tool.shutdown_vm.assert_called_once()

    def test_start_vm(self, hv):
        hv.start_vm("VM_Name")
        hv.virt_tool.start_vm.assert_called_once()

    def test_delete_vm(self, hv):
        hv.delete_vm("VM_Name")
        hv.virt_tool.delete_vm.assert_called_once()

    def test_wait_for_vm_state(self, hv, caplog, mocker):
        caplog.set_level(log_levels.MODULE_DEBUG)
        mocker.patch("mfd_kvm.hypervisor.sleep")
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": "some_state"})
        assert hv.wait_for_vm_state("VM_name", "some_state")

    def test_wait_for_vm_timeout(self, hv, caplog, mocker):
        caplog.set_level(log_levels.MODULE_DEBUG)
        mocker.patch("mfd_kvm.hypervisor.sleep")
        mocker.patch("mfd_kvm.hypervisor.time", side_effect=(60, 61, 90, 120))
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": "wrong_state"})
        assert not hv.wait_for_vm_state("VM_name", "some_state")

    def test_wait_for_vm_down(self, hv, caplog, mocker):
        caplog.set_level(log_levels.MODULE_DEBUG)
        mocker.patch("mfd_kvm.hypervisor.sleep")
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": "shut off"})
        assert hv.wait_for_vm_down("VM_name")

    def test_wait_for_vm_up(self, hv, caplog, mocker):
        caplog.set_level(log_levels.MODULE_DEBUG)
        mocker.patch("mfd_kvm.hypervisor.sleep")
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": "running"})
        assert hv.wait_for_vm_up("VM_name")

    def test_stop_all_vms(self, hv, mocker):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        mocker.patch("mfd_kvm.KVMHypervisor.wait_for_vm_state", return_value=True)
        assert hv.stop_all_vms()
        assert hv.virt_tool.shutdown_gracefully_vm.call_count == 2

    def test_stop_all_vms_force(self, hv, mocker):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        mocker.patch("mfd_kvm.KVMHypervisor.wait_for_vm_state", return_value=True)
        assert hv.stop_all_vms(True)
        assert hv.virt_tool.shutdown_vm.call_count == 2

    def test_stop_all_vms_fails(self, hv, mocker):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        mocker.patch("mfd_kvm.KVMHypervisor.wait_for_vm_state", return_value=False)
        assert not hv.stop_all_vms()
        assert hv.virt_tool.shutdown_gracefully_vm.call_count == 2

    def test_start_all_vms(self, hv, mocker):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        mocker.patch("mfd_kvm.KVMHypervisor.wait_for_vm_state", return_value=True)
        assert hv.start_all_vms()
        assert hv.virt_tool.start_vm.call_count == 2

    def test_start_all_vms_fails(self, hv, mocker):
        hv.virt_tool.list_vms.return_value = [
            {"name": "foo-055-045"},
            {"name": "Base_R82.img_VM001_10.10.10.11"},
        ]
        mocker.patch("mfd_kvm.KVMHypervisor.wait_for_vm_state", return_value=False)
        assert not hv.start_all_vms()
        assert hv.virt_tool.start_vm.call_count == 1

    def test_get_vfs_id_for_pf(self, hv):
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn0 -> ../0000:18:10.1
        lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn1 -> ../0000:18:10.3
        lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn2 -> ../0000:18:10.5
        lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn3 -> ../0000:18:10.7
        lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn25 -> ../0000:18:10.19"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_vfs_id_for_pf(interface="eth1") == [0, 1, 2, 3, 25]

    def test_get_vfs_id_for_pf_not_found(self, hv):
        output = "broken output"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        with pytest.raises(VFExceptionKVM):
            hv.get_vfs_id_for_pf(interface="eth1")

    def test_get_vfs_id_for_pf_failure(self, hv):
        output = "ls: cannot access '/sys/class/net/eth1/device/virtfn*': No such file or directory"
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output=output)
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_vfs_id_for_pf(interface="eth1")

    def test_get_pci_address_for_vf(self, hv):
        output = "lrwxrwxrwx 1 root root 0 Jan 27 13:53 /sys/class/net/eth1/device/virtfn0 -> ../0000:18:10.1"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_pci_address_for_vf(interface="eth1", vf_id=0) == PCIAddress(0, 24, 16, 1)

    def test_get_pci_address_for_vf_not_found(self, hv):
        output = "broken output"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        with pytest.raises(VFExceptionKVM):
            hv.get_pci_address_for_vf(interface="eth1", vf_id=0)

    def test_get_pci_address_for_vf_failure(self, hv):
        output = "ls: cannot access '/sys/class/net/eth1/device/virtfn0': No such file or directory"
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output=output)
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_pci_address_for_vf(interface="eth1", vf_id=0)

    def test_set_number_of_vfs_for_pf_interface_not_found(self, hv):
        output = "ls: cannot access '/sys/class/net/eth1/': No such file or directory"
        hv._conn.execute_command.side_effect = NotFoundInterfaceKVM(returncode=1, cmd="", output=output)
        with pytest.raises(NotFoundInterfaceKVM):
            hv.set_number_of_vfs_for_pf(interface="eth1", vfs_count=4, check=False)

    def test_get_pci_address_for_vf_by_pci(self, hv):
        output = "lrwxrwxrwx 1 root root 0 Jan 31 14:00 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:18:10.1"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_pci_address_for_vf_by_pci(pf_pci_address=PCIAddress(data="0000:5e:00.0"), vf_id=0) == PCIAddress(
            0, 24, 16, 1
        )

    def test_get_pci_address_for_vf_by_pci_failure(self, hv):
        output = "ls: cannot access '/sys/bus/pci/devices/0000:5e:00.0/virtfn0': No such file or directory"
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output=output)
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_pci_address_for_vf_by_pci(pf_pci_address=PCIAddress(data="0000:5e:00.0"), vf_id=0)

    def test_set_number_of_vfs_for_pf(self, hv, mocker):
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # check if interface available
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # set vf count
        ]
        hv.set_number_of_vfs_for_pf(interface="eth1", vfs_count=4, check=False)
        calls = [
            mocker.call("ls /sys/class/net/eth1", custom_exception=NotFoundInterfaceKVM),
            mocker.call(
                "echo 4 > /sys/class/net/eth1/device/sriov_numvfs",
                shell=True,
                expected_return_codes=[0, 1],
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
        ]
        hv._conn.execute_command.assert_has_calls(calls)

    def test_check_number_of_vfs(self, hv):
        output = dedent(
            """\
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn0 -> ../0000:18:10.1
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn1 -> ../0000:18:10.3
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn2 -> ../0000:18:10.5
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn3 -> ../0000:18:10.7
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn25 -> ../0000:18:10.19
            """
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout=output, stderr="", return_code=0
        )
        hv.check_number_of_vfs(interface="eth1", vfs_count=5)

    def test_check_number_of_vfs_no_vfs_present(self, hv):
        output = "ls: cannot access /sys/class/net/eth1/device/virtfn*: No such file or directory"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stderr=output, return_code=2)
        hv.check_number_of_vfs(interface="eth1", vfs_count=0)

    def test_check_number_of_vfs_incorrect(self, hv):
        output = dedent(
            """\
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn0 -> ../0000:18:10.1
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn1 -> ../0000:18:10.3
            lrwxrwxrwx 1 root root 0 Aug 11 13:01 /sys/class/net/eth1/device/virtfn2 -> ../0000:18:10.5"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout=output, stderr="", return_code=0
        )
        with pytest.raises(VFExceptionKVM):
            hv.check_number_of_vfs(interface="eth1", vfs_count=4)

    def test_set_number_of_vfs_for_pf_already_configured_flow(self, hv, mocker):
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # check if interface available
            ConnectionCompletedProcess(
                args="",
                stdout="",
                stderr="echo: write error: Device or resource busy",
                return_code=1,
            ),  # try set vf count
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # disable sriov
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # set vf count
        ]
        hv.set_number_of_vfs_for_pf(interface="eth1", vfs_count=4, check=False)
        calls = [
            mocker.call("ls /sys/class/net/eth1", custom_exception=NotFoundInterfaceKVM),
            mocker.call(
                "echo 4 > /sys/class/net/eth1/device/sriov_numvfs",
                shell=True,
                expected_return_codes=[0, 1],
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
            mocker.call(
                "echo 0 > /sys/class/net/eth1/device/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
            mocker.call(
                "echo 4 > /sys/class/net/eth1/device/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
        ]
        hv._conn.execute_command.assert_has_calls(calls)

    def test_set_number_of_vfs_for_pf_incorrect_count(self, hv, mocker):
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # check if interface available
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # set vf count
        ]
        hv.check_number_of_vfs = mocker.create_autospec(
            hv.check_number_of_vfs,
            side_effect=VFExceptionKVM("Mismatched count of expected and created VFs 3 != 4"),
        )
        with pytest.raises(VFExceptionKVM, match="Mismatched count of expected and created VFs 3 != 4"):
            hv.set_number_of_vfs_for_pf(interface="eth1", vfs_count=4)

    def test_prepare_vf_xml(self, hv, mocker):
        template_data = b"domain='{{domain}}' bus='{{bus}}' slot='{{slot}}' function='{{func}}"
        mocker.patch("mfd_kvm.hypervisor.open", mocker.mock_open(read_data=template_data))
        hv._conn.path = mocker.Mock()
        hv.prepare_vf_xml(template_path="", file_to_save="", pci_address=PCIAddress(0, 0, 15, 1))
        hv._conn.path().write_text.assert_called_once()

    def test_prepare_pci_controller_xml(self, hv, mocker):
        template_data = (
            b"index='{{index}}' chassis='{{chassis}}' port='{{port}}' bus='{{bus}}' slot='{{slot}}' "
            b"function='{{func}}"
        )
        mocker.patch("mfd_kvm.hypervisor.open", mocker.mock_open(read_data=template_data))
        hv._conn.path = mocker.Mock()
        hv.prepare_pci_controller_xml(
            template_path="",
            file_to_save="",
            pci_address=PCIAddress(0, 0, 15, 1),
            index=1,
            port=0x1F,
            chassis=0x1F,
        )
        hv._conn.path().write_text.assert_called_once()

    def test_prepare_pci_controller_xml_check_input_to_xml(self, hv, mocker):
        template_data = (
            b"index='{{index}}' chassis='{{chassis}}' port='{{port}}' bus='{{bus}}' slot='{{slot}}' "
            b"function='{{func}}"
        )
        mocker.patch("mfd_kvm.hypervisor.open", mocker.mock_open(read_data=template_data))
        hv._render_file = mocker.create_autospec(hv._render_file)
        hv.prepare_pci_controller_xml(
            template_path="",
            file_to_save="",
            pci_address=PCIAddress(0, 0, 15, 1),
            index=1,
            port=0x1F,
            chassis=0x1F,
        )
        expected_data = {
            "domain": "0x0",
            "bus": "0x0",
            "slot": "0xf",
            "func": "0x1",
            "index": 1,
            "chassis": 31,
            "port": "0x1f",
        }
        hv._render_file.assert_called_with("", expected_data, "")

    def test_detach_vf(self, hv, mocker):
        mocker.patch("mfd_kvm.KVMHypervisor.detach_device")
        hv.detach_vf(name="VM_name", vf_config="/path/to/config/file")
        hv.detach_device.assert_called_once()

    @pytest.mark.parametrize("state", ["shut off", "running"])
    def test_detach_device(self, hv, mocker, state):
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": state})
        hv.detach_device(name="VM_name", device_config="/path/to/config/file")
        hv.virt_tool.detach_device.assert_called_once()

    def test_attach_vf(self, hv, mocker):
        mocker.patch("mfd_kvm.KVMHypervisor.attach_device")
        hv.attach_vf(name="VM_name", vf_config="/path/to/config/file")
        hv.attach_device.assert_called_once()

    def test_attach_agent(self, hv, mocker):
        mocker.patch("mfd_kvm.KVMHypervisor.attach_device")
        hv.attach_agent(name="VM_name", agent_config_file="/path/to/config/file")
        hv.attach_device.assert_called_once()

    @pytest.mark.parametrize("state", ["shut off", "running"])
    def test_attach_device(self, hv, mocker, state):
        mocker.patch("mfd_kvm.KVMHypervisor.get_vm_status", return_value={"State": state})
        hv.attach_device(name="VM_name", device_config="/path/to/config/file")
        hv.virt_tool.attach_device.assert_called_once()

    def test_clone_vm_hdd_image_check_if_source_file_exists(self, hv):
        source_path = Path("/foo/source")
        dest_path = Path("/foo/destination")
        hv._conn.path.side_effect = Path
        escaped_source_path = re.escape(str(source_path))
        with pytest.raises(FileNotFoundError, match=f"Not found {escaped_source_path} in system."):
            hv.clone_vm_hdd_image(path_to_source_image=source_path, path_to_destination_image=dest_path)

    def test_clone_vm_hdd_image_checks_destination_on_connection(self, hv, mocker, tmp_path):
        source_path = tmp_path / "source.img"
        source_path.touch()
        destination_path = tmp_path / "destination.img"
        destination_path.touch()
        remote_source_path = mocker.MagicMock()
        remote_source_path.__str__.return_value = "/remote/source.img"
        remote_source_path.is_file.return_value = True
        remote_destination_path = mocker.MagicMock()
        remote_destination_path.__str__.return_value = "/remote/destination.img"
        remote_destination_path.exists.return_value = False
        hv._conn.path.side_effect = [remote_source_path, remote_destination_path]
        hv.__dict__["_rsync_available"] = True
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="6", return_code=0)
        process = mocker.create_autospec(RPyCProcess)
        process.running = False
        hv._conn.start_process.return_value = process

        result = hv.clone_vm_hdd_image(
            path_to_source_image=source_path,
            path_to_destination_image=destination_path,
        )

        assert result == remote_destination_path
        assert hv._conn.path.call_args_list == [mocker.call(str(source_path)), mocker.call(str(destination_path))]
        hv._conn.execute_command.assert_called_once_with(
            "ls /remote/source.img -l | awk '{print $5}'",
            shell=True,
        )
        hv._conn.start_process.assert_called_once_with("rsync -aq /remote/source.img /remote/destination.img")

    def test_clone_vm_hdd_image_calculated_checksums_match(self, hv, mocker, tmp_path):
        source_path = tmp_path / "source.img"
        source_path.touch()
        destination_path = tmp_path / "destination.img"
        destination_path.touch()
        hv._conn.path.side_effect = Path
        checksum = "d41d8cd98f00b204e9800998ecf8427e"
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", stdout=f"{checksum}  {source_path}", return_code=0),
            ConnectionCompletedProcess(args="", stdout=f"{checksum}  {destination_path}", return_code=0),
        ]

        result = hv.clone_vm_hdd_image(
            path_to_source_image=source_path,
            path_to_destination_image=destination_path,
        )

        assert result == destination_path
        assert hv._conn.execute_command.call_args_list == [
            mocker.call(f"md5sum {source_path}"),
            mocker.call(f"md5sum {destination_path}"),
        ]
        assert hv._conn.path.call_args_list == [mocker.call(str(source_path)), mocker.call(str(destination_path))]
        hv._conn.start_process.assert_not_called()

    def test_clone_vm_hdd_image_file_checksum_matches(self, hv, mocker, tmp_path):
        source_path = tmp_path / "source.img"
        source_path.touch()
        destination_path = tmp_path / "destination.img"
        destination_path.touch()
        hv._conn.path.side_effect = Path
        checksum = "d41d8cd98f00b204e9800998ecf8427e"
        source_path.with_name(f"{source_path.name}.md5").write_text(f"{checksum.upper()}  {source_path.name}\n")
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout=f"{checksum}  {destination_path}", return_code=0
        )

        result = hv.clone_vm_hdd_image(
            path_to_source_image=source_path,
            path_to_destination_image=destination_path,
        )

        assert result == destination_path
        hv._conn.execute_command.assert_called_once_with(f"md5sum {destination_path}")
        hv._conn.start_process.assert_not_called()

    def test_clone_vm_hdd_image_file_checksum_differs(self, hv, mocker, tmp_path, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        timeout_mocker = mocker.patch("mfd_kvm.hypervisor.TimeoutCounter")
        timeout_mocker.return_value.__bool__.return_value = False
        process = mocker.create_autospec(RPyCProcess)
        process.running = False
        hv._conn.start_process.return_value = process
        source_path = tmp_path / "source.img"
        source_path.write_text("source")
        destination_path = tmp_path / "destination.img"
        destination_path.write_text("destination")
        hv._conn.path.side_effect = Path
        source_checksum = "36cd38f49b9afa08222c0dc9ebfe35eb"
        destination_checksum = "cc17cbb4f2b32c03a280a68544e8516f"
        source_path.with_name(f"{source_path.name}.md5").write_text(source_checksum)
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", stdout=f"{destination_checksum}  {destination_path}", return_code=0),
            ConnectionCompletedProcess(args="", return_code=0),
            ConnectionCompletedProcess(args="", stdout="6", return_code=0),
        ]

        result = hv.clone_vm_hdd_image(
            path_to_source_image=source_path,
            path_to_destination_image=destination_path,
        )

        assert result == destination_path
        assert not destination_path.exists()
        assert f"Destination image {destination_path} differs from the source; removing it." in caplog.text
        hv._conn.start_process.assert_called_once_with(f"rsync -aq {source_path} {destination_path}")

    def test_clone_vm_hdd_image_timeout_exceeded(self, hv, mocker):
        timeout_mocker = mocker.patch("mfd_kvm.hypervisor.TimeoutCounter")
        timeout_mocker.return_value.__bool__.return_value = True
        timeout = 12345
        dest_path = Path("/foo/destination")
        mock_path = mocker.patch("mfd_kvm.hypervisor.Path")
        hv._conn.path.side_effect = [mock_path, dest_path]
        escaped_source_path = re.escape(str(mock_path))
        mock_path.is_file.return_value = True
        with pytest.raises(
            KVMHypervisorException,
            match=f"Cloning image {escaped_source_path} not finished in given timeout: {timeout}",
        ):
            hv.clone_vm_hdd_image(
                path_to_source_image=mock_path,
                path_to_destination_image=dest_path,
                timeout=timeout,
            )

    @pytest.mark.parametrize(
        ("rsync_return_code", "copy_command"),
        [(0, "rsync -aq"), (1, "scp")],
    )
    def test_clone_vm_hdd_image_source_file_exists_and_succeeded(self, hv, mocker, rsync_return_code, copy_command):
        timeout_mocker = mocker.patch("mfd_kvm.hypervisor.TimeoutCounter")
        timeout_mocker.return_value.__bool__.return_value = False
        process = mocker.create_autospec(RPyCProcess)
        process.running = False
        hv._conn.start_process.return_value = process
        dest_path = Path("/foo/destination")
        path_mocker = mocker.patch("mfd_kvm.hypervisor.Path")
        path_mocker.is_file.return_value = True
        hv._conn.path.side_effect = [path_mocker, dest_path, path_mocker, dest_path]
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", return_code=rsync_return_code),
            ConnectionCompletedProcess(args="", stdout="100", return_code=0),
            ConnectionCompletedProcess(args="", stdout="100", return_code=0),
        ]
        new_names = [
            hv.clone_vm_hdd_image(path_to_source_image=path_mocker, path_to_destination_image=dest_path)
            for _ in range(2)
        ]

        assert hv._conn.execute_command.call_args_list == [
            mocker.call("command -v rsync", shell=True, expected_return_codes={0, 1}),
            mocker.call(f"ls {path_mocker} -l | awk '{{print $5}}'", shell=True),
            mocker.call(f"ls {path_mocker} -l | awk '{{print $5}}'", shell=True),
        ]
        assert hv._conn.start_process.call_args_list == [
            mocker.call(f"{copy_command} {path_mocker} {dest_path}"),
            mocker.call(f"{copy_command} {path_mocker} {dest_path}"),
        ]
        assert new_names == [dest_path, dest_path]

    def test_clone_vm_hdd_image_still_cloning(self, hv, mocker, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        timeout_mocker = mocker.patch("mfd_kvm.hypervisor.TimeoutCounter")
        timeout_mocker.return_value.__bool__.return_value = False
        process = mocker.create_autospec(RPyCProcess)
        type(process).running = mocker.PropertyMock(side_effect=[True, False])
        hv._conn.start_process.return_value = process
        path_mocker = mocker.patch("mfd_kvm.hypervisor.Path")
        dest_path = mocker.patch("mfd_kvm.hypervisor.Path")
        mocker.patch("mfd_kvm.hypervisor.sleep")
        path_mocker.is_file.return_value = True
        dest_path.exists.side_effect = [False, True]
        hv._conn.path.side_effect = [path_mocker, dest_path]
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", return_code=0),
            ConnectionCompletedProcess(args="", stdout="100", return_code=0),
            ConnectionCompletedProcess(args="", stdout="10", return_code=0),
        ]
        hv.clone_vm_hdd_image(path_to_source_image=path_mocker, path_to_destination_image=dest_path)
        assert "still cloning... 10 %, next check in 30secs." in caplog.text

    @pytest.mark.parametrize("current_size", ["", "invalid", "0", "-1"])
    def test_clone_vm_hdd_image_invalid_current_size(self, hv, mocker, caplog, current_size):
        caplog.set_level(log_levels.MODULE_DEBUG)
        timeout_mocker = mocker.patch("mfd_kvm.hypervisor.TimeoutCounter")
        timeout_mocker.return_value.__bool__.return_value = False
        process = mocker.create_autospec(RPyCProcess)
        type(process).running = mocker.PropertyMock(side_effect=[True, False])
        hv._conn.start_process.return_value = process
        path_mocker = mocker.patch("mfd_kvm.hypervisor.Path")
        dest_path = mocker.patch("mfd_kvm.hypervisor.Path")
        mocker.patch("mfd_kvm.hypervisor.sleep")
        path_mocker.is_file.return_value = True
        dest_path.exists.side_effect = [False, True]
        hv._conn.path.side_effect = [path_mocker, dest_path]
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(args="", return_code=0),
            ConnectionCompletedProcess(args="", stdout="100", return_code=0),
            ConnectionCompletedProcess(args="", stdout=current_size, return_code=0),
        ]

        hv.clone_vm_hdd_image(path_to_source_image=path_mocker, path_to_destination_image=dest_path)

        assert "still cloning..." not in caplog.text

    def test_create_mdev(self, hv, mocker):
        command = r'echo "a1234" | tee /sys/class/mdev_bus/0000\:b9\:00.0/mdev_supported_types/ice-vdcm/create'
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="a1234", return_code=0)
        hv._render_file = mocker.create_autospec(hv._render_file, return_value={"mdev.xml"})
        hv.create_mdev(
            mdev_uuid="a1234",
            pci_address=PCIAddress(domain=0, bus=185, slot=0, func=0),
            file_to_save="mdev.xml",
        )

        hv._conn.execute_command.assert_called_once_with(
            command, custom_exception=KVMHypervisorExecutionException, shell=True
        )

    def test_create_mdev_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.create_mdev(
                mdev_uuid="a1234",
                pci_address=PCIAddress(domain=0, bus=185, slot=0, func=0),
                file_to_save="mdev.xml",
            )

    def test_create_mdev_failure_no_mdev_uuid_in_output(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="", return_code=0)
        with pytest.raises(KVMHypervisorException, match="a1234 not found in cmd output: "):
            hv.create_mdev(
                mdev_uuid="a1234",
                pci_address=PCIAddress(domain=0, bus=185, slot=0, func=0),
                file_to_save="mdev.xml",
            )

    def test_destroy_mdev(self, hv):
        command = "echo 1 > /sys/bus/mdev/devices/a1234/remove"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        hv.destroy_mdev(
            mdev_uuid="a1234",
        )
        hv._conn.execute_command.assert_called_once_with(
            command, custom_exception=KVMHypervisorExecutionException, shell=True
        )

    def test_destroy_mdev_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.destroy_mdev(mdev_uuid="a1234")

    def test_get_hdd_path(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <source file='/var/lib/libvirt/images/gklab-55-034.qcow2' index='1'/>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        assert hv.get_hdd_path("domain") == hv._conn.path("/var/lib/libvirt/images/gklab-55-034.qcow2")

    def test_get_hdd_path_no_source_file(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <driver name='qemu' type='qcow2'/>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="HDD path for domain not found in dumped xml!"):
            hv.get_hdd_path("domain")

    def test_get_hdd_path_no_file_attribute(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <source index='1'/>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="HDD path for domain not found in dumped xml!"):
            hv.get_hdd_path("domain")

    def test_dump_xml(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <interface type='hostdev' managed='yes'>
                <mac address='aa:bb:cc:de:ed:be'/>
                <driver name='vfio'/>
                <source>
                    <address type='pci' domain='0x0000' bus='0x5e' slot='0x11' function='0x1'/>
                </source>
                <link state='up'/>
                <alias name='hostdev1'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0'/>
            </interface>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        assert isinstance(hv.dump_xml("vm_name"), ET.ElementTree)
        hv.virt_tool.dump_xml.assert_called_once()

    def test_get_pci_for_host_vf_and_vm_vf(self, hv, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        name = "domain"
        expected_host_pci, expected_vm_pci = (
            PCIAddress(0, 94, 17, 1),
            PCIAddress(0, 0, 8, 0),
        )
        log_message = f"VM: {name}, Host VF PCI: {expected_host_pci}, VM VF PCI: {expected_vm_pci}"
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <hostdev mode='subsystem' type='pci' managed='yes'>
                <driver name='vfio'/>
                <source>
                    <address domain='0x0000' bus='0x5e' slot='0x11' function='0x1'/>
                </source>
                <alias name='hostdev0'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0'/>
            </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        assert hv.get_pci_for_host_vf_and_vm_vf(name) == [(expected_host_pci, expected_vm_pci)]
        assert log_message in caplog.messages

    def test_get_pci_for_host_vf_and_vm_vf_no_interface(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <memory unit='KiB'>2097152</memory>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(
            KVMHypervisorException,
            match="Interface with Host VF and VM VF not found in xml!",
        ):
            hv.get_pci_for_host_vf_and_vm_vf("domain")

    def test_get_pci_for_host_vf_and_vm_vf_no_tag(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <hostdev mode='subsystem' type='pci' managed='yes'>
                <driver name='vfio'/>
                <alias name='hostdev0'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0'/>
            </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="PCI not found in xml!"):
            hv.get_pci_for_host_vf_and_vm_vf("domain")

    def test_get_pci_for_host_vf_and_vm_vf_no_attribute(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <hostdev mode='subsystem' type='pci' managed='yes'>
                <driver name='vfio'/>
                <source>
                  <address/>
                </source>
                <alias name='hostdev0'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0'/>
            </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="PCI not found in xml!"):
            hv.get_pci_for_host_vf_and_vm_vf("domain")

    def test_get_pci_addresses_of_vfs(self, hv):
        expected_output = [
            PCIAddress(domain=0, bus=94, slot=2, func=0),
            PCIAddress(domain=0, bus=94, slot=2, func=1),
            PCIAddress(domain=0, bus=94, slot=2, func=2),
            PCIAddress(domain=0, bus=94, slot=2, func=3),
        ]
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/class/net/eth2/device/virtfn0 -> ../0000:5e:02.0
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/class/net/eth2/device/virtfn1 -> ../0000:5e:02.1
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/class/net/eth2/device/virtfn2 -> ../0000:5e:02.2
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/class/net/eth2/device/virtfn3 -> ../0000:5e:02.3"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_pci_addresses_of_vfs(interface="eth2") == expected_output

    def test_get_pci_addresses_of_vfs_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_pci_addresses_of_vfs(interface="eth2")

    def test_get_pci_addresses_of_vfs_by_pci(self, hv):
        expected_output = [
            PCIAddress(domain=0, bus=94, slot=2, func=0),
            PCIAddress(domain=0, bus=94, slot=2, func=1),
            PCIAddress(domain=0, bus=94, slot=2, func=2),
            PCIAddress(domain=0, bus=94, slot=2, func=3),
        ]
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:5e:02.0
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:5e:02.1
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:5e:02.2
        lrwxrwxrwx 1 root root 0 Aug  5 10:56 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:5e:02.3"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_pci_addresses_of_vfs_by_pci(pci_address=PCIAddress(data="0000:5e:00.0")) == expected_output

    def test_get_pci_addresses_of_vfs_by_pci_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_pci_addresses_of_vfs_by_pci(pci_address=PCIAddress(data="0000:5e:00.0"))

    def test_get_vf_id_from_pci(self, hv):
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/class/net/eth2/device/virtfn0 -> ../0000:5e:0a.0
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/class/net/eth2/device/virtfn1 -> ../0000:5e:0a.1
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/class/net/eth2/device/virtfn25 -> ../0000:5e:0a.19"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_vf_id_from_pci(interface="eth2", pci=PCIAddress(domain=0, bus=94, slot=10, func=25)) == 25

    def test_get_vf_id_from_pci_command_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=2, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_vf_id_from_pci(interface="eth2", pci=PCIAddress(domain=0, bus=94, slot=10, func=0))

    def test_get_vf_id_from_pci_no_matched_vf(self, hv):
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/class/net/eth2/device/virtfn1 -> ../0000:5e:0a.1"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        with pytest.raises(VFExceptionKVM, match="Not matched VFs for interface eth2."):
            hv.get_vf_id_from_pci(interface="eth2", pci=PCIAddress(domain=0, bus=94, slot=10, func=0))

    def test_get_vf_id_by_pci(self, hv):
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/bus/pci/devices/0000:5e:00.0/virtfn0 -> ../0000:5e:0a.0
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/bus/pci/devices/0000:5e:00.0/virtfn1 -> ../0000:5e:0a.1
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/bus/pci/devices/0000:5e:00.0/virtfn25 -> ../0000:5e:0a.19"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert (
            hv.get_vf_id_by_pci(
                pf_pci_address=PCIAddress(data="0000:5e:00.0"),
                vf_pci_address=PCIAddress(data="0000:5e:0a.19"),
            )
            == 25
        )

    def test_get_vf_id_by_pci_command_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=2, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_vf_id_by_pci(
                pf_pci_address=PCIAddress(data="0000:5e:00.0"),
                vf_pci_address=PCIAddress(data="0000:5e:0a.0"),
            )

    def test_get_vf_id_by_pci_no_matched_vf(self, hv):
        output = dedent(
            """\
        lrwxrwxrwx 1 root root 0 Sep  8 13:58 /sys/bus/pci/devices/0000:5e:00.0/virtfn1 -> ../0000:5e:0a.1"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        with pytest.raises(VFExceptionKVM, match="Not matched VFs for PF PCI Address 0000:5e:00.0"):
            hv.get_vf_id_by_pci(
                pf_pci_address=PCIAddress(data="0000:5e:00.0"),
                vf_pci_address=PCIAddress(data="0000:5e:0a.0"),
            )

    def test_create_bridge(self, hv):
        command = "brctl addbr test_bridge"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        hv.create_bridge(bridge_name="test_bridge")
        hv._conn.execute_command.assert_called_once_with(
            command, custom_exception=KVMHypervisorExecutionException, shell=True
        )

    def test_create_bridge_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.create_bridge(bridge_name="test_bridge")

    def test_delete_bridge(self, hv):
        command = "brctl delbr test_bridge"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        hv.delete_bridge(bridge_name="test_bridge")
        hv._conn.execute_command.assert_called_once_with(
            command, custom_exception=KVMHypervisorExecutionException, shell=True
        )

    def test_delete_bridge_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.delete_bridge(bridge_name="test_bridge")

    def test_add_interface_to_bridge(self, hv):
        command = "brctl addif test_bridge test_interface"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", return_code=0)
        hv.add_interface_to_bridge(bridge_name="test_bridge", interface="test_interface")
        hv._conn.execute_command.assert_called_once_with(
            command, custom_exception=KVMHypervisorExecutionException, shell=True
        )

    def test_add_interface_to_bridge_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.add_interface_to_bridge(bridge_name="test_bridge", interface="test_interface")

    def test_detach_interfaces(self, hv, mocker, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        hv._conn.path.cwd = mocker.Mock()
        hv.get_pci_for_host_vf_and_vm_vf = mocker.Mock()
        hv.get_pci_for_host_vf_and_vm_vf.return_value = [(PCIAddress(0, 94, 17, 1), PCIAddress(0, 0, 8, 0))]
        hv.prepare_vf_xml = mocker.create_autospec(hv.prepare_vf_xml)
        hv.detach_vf = mocker.create_autospec(hv.detach_vf)

        hv.detach_interfaces(["vm_1"])

        expected_message = "Interface 0000:00:08.0 detached from vm_1"
        assert expected_message in caplog.text

    def test_detach_interfaces_detach_failure(self, hv, mocker, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        hv._conn.path.cwd = mocker.Mock()
        hv.get_pci_for_host_vf_and_vm_vf = mocker.Mock()
        hv.get_pci_for_host_vf_and_vm_vf.return_value = [(PCIAddress(0, 94, 17, 1), PCIAddress(0, 0, 8, 0))]
        hv.prepare_vf_xml = mocker.create_autospec(hv.prepare_vf_xml)
        hv.detach_vf = mocker.Mock()
        hv.detach_vf.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")

        hv.detach_interfaces(["vm_1"])

        expected_message = "Interface 0000:00:08.0 couldn't be detached from vm_1: "
        assert expected_message in caplog.text

    def test_get_dynamic_ram(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="18000", return_code=0)
        assert hv.get_dynamic_ram(vm_number=2) == 4000

    def test_get_dynamic_ram_free_more_than_max(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="40000", return_code=0)
        assert hv.get_dynamic_ram(vm_number=2) == 10000

    def test_get_dynamic_ram_no_output_from_awk_command(self, hv, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="", return_code=1)
        assert hv.get_dynamic_ram(vm_number=2) == 2000
        assert "There's not output from awk, proceeding with default 2000 MB" in caplog.text

    def test_get_dynamic_ram_free_less_then_reserved_and_minimal(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="13000", return_code=0)
        assert hv.get_dynamic_ram(vm_number=2) == 2000

    def test_get_dynamic_ram_not_enough_free_ram(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="3000", return_code=0)
        with pytest.raises(KVMHypervisorException, match="Not enough free RAM on SUT for VM."):
            hv.get_dynamic_ram(vm_number=2)

    def test_get_mdev_details(self, hv, caplog):
        caplog.set_level(log_levels.MODULE_DEBUG)
        name = "domain"
        (
            expected_mdev_uuid,
            expected_vm_pci,
        ) = "d08c195c-bddb-4fbb-b7aa-0fa10a96c5b1", PCIAddress(0, 0, 5, 0)
        log_message = f"VM: {name}, Host UUID: {expected_mdev_uuid}, VM VF PCI: {expected_vm_pci}"
        output = dedent(
            """\
        <domain type='kvm' id='1'>
                <hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci' display='off'>
               <source>
                 <address uuid='d08c195c-bddb-4fbb-b7aa-0fa10a96c5b1'/>
               </source>
               <alias name='hostdev0'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0'/>
                </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        assert hv.get_mdev_details(name) == [(expected_mdev_uuid, expected_vm_pci)]
        assert log_message in caplog.messages

    def test_get_mdev_details_no_interface(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
            <memory unit='KiB'>2097152</memory>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="Interface with MDEV not found in xml!"):
            hv.get_mdev_details("domain")

    def test_get_mdev_details_no_tag(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
                <hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci' display='off'>
               <source>
                 <address/>
               </source>
               <alias name='hostdev0'/>
                <address type='pci' domain='0x0000' bus='0x00' slot='0x05' function='0x0'/>
                </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="Interface with MDEV does not contains UUID!"):
            hv.get_mdev_details("domain")

    def test_get_mdev_details_no_attribute(self, hv):
        output = dedent(
            """\
        <domain type='kvm' id='1'>
                <hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci' display='off'>
               <source>
                 <address uuid='d08c195c-bddb-4fbb-b7aa-0fa10a96c5b1'/>
               </source>
               <alias name='hostdev0'/>
                <address type='pci' bus='0x00' slot='0x05' function='0x0'/>
                </hostdev>
        </domain>"""
        )
        hv.virt_tool.dump_xml.return_value = ET.ElementTree(ET.fromstring(output))
        with pytest.raises(KVMHypervisorException, match="PCI not found in xml!"):
            hv.get_mdev_details("domain")

    def test_get_pci_address_of_mdev_pf(self, hv):
        output = "0000:5E:00.0"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_pci_address_of_mdev_pf(mdev_uuid="a1234") == PCIAddress(0, 94, 0, 0)

    def test_get_pci_address_of_mdev_pf_for_vf_no_cmd_output(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="", return_code=0)
        with pytest.raises(VFExceptionKVM, match="Not matched PF PCI for MDEV with UUID: a1234"):
            hv.get_pci_address_of_mdev_pf(mdev_uuid="a1234")

    def test_get_pci_address_of_mdev_pf_vf_command_failure(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(returncode=1, cmd="", output="")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_pci_address_of_mdev_pf(mdev_uuid="a1234")

    def test_get_all_mdev_uuids(self, hv):
        output = "97cef11e-ebaa-44a5-bf31-41340117e172  ea684326-0dce-43da-8a5a-6e61740fc2e0"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_all_mdev_uuids() == [
            "97cef11e-ebaa-44a5-bf31-41340117e172",
            "ea684326-0dce-43da-8a5a-6e61740fc2e0",
        ]

    def test_get_all_mdev_uuids_no_invalid_output(self, hv):
        output = "Invalid output"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        with pytest.raises(KVMHypervisorException, match="MDEV UUIDs not found!: "):
            hv.get_all_mdev_uuids()

    def test_set_number_of_vfs_for_pf_by_pci(self, hv):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout="", return_code=0)
        hv.set_number_of_vfs_for_pf_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=7, check=False)
        hv._conn.execute_command.assert_called_once_with(
            "echo 7 > /sys/bus/pci/devices/0000:18:10.1/sriov_numvfs",
            custom_exception=KVMHypervisorExecutionException,
            shell=True,
            expected_return_codes={0, 1},
            timeout=60,
        )

    def test_set_number_of_vfs_for_pf_by_pci_already_configured_flow(self, hv, mocker):
        hv._conn.execute_command.side_effect = [
            ConnectionCompletedProcess(
                args="",
                stdout="",
                stderr="echo: write error: Device or resource busy",
                return_code=1,
            ),  # try set vf count
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # disable sriov
            ConnectionCompletedProcess(args="", stdout="", return_code=0),  # set vf count
        ]
        hv.set_number_of_vfs_for_pf_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=7, check=False)
        calls = [
            mocker.call(
                "echo 7 > /sys/bus/pci/devices/0000:18:10.1/sriov_numvfs",
                shell=True,
                expected_return_codes={0, 1},
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
            mocker.call(
                "echo 0 > /sys/bus/pci/devices/0000:18:10.1/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
            mocker.call(
                "echo 7 > /sys/bus/pci/devices/0000:18:10.1/sriov_numvfs",
                shell=True,
                custom_exception=KVMHypervisorExecutionException,
                timeout=60,
            ),
        ]
        hv._conn.execute_command.assert_has_calls(calls)

    def test_set_number_of_vfs_for_pf_by_pci_incorrect_count(self, hv, mocker):
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout="", return_code=0
        )  # set vf count
        hv.check_number_of_vfs_by_pci = mocker.create_autospec(
            hv.check_number_of_vfs_by_pci,
            side_effect=VFExceptionKVM("Mismatched count of expected and created VFs 4 != 7"),
        )
        with pytest.raises(VFExceptionKVM, match="Mismatched count of expected and created VFs 4 != 7"):
            hv.set_number_of_vfs_for_pf_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=7)

    def test_check_number_of_vfs_by_pci(self, hv):
        output = dedent(
            """\
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn0 -> ../0000:af:02.0
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn1 -> ../0000:af:02.1
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn2 -> ../0000:af:02.2
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn3 -> ../0000:af:02.3
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn4 -> ../0000:af:02.4
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn5 -> ../0000:af:02.5
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn6 -> ../0000:af:02.6
            """
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout=output, stderr="", return_code=0
        )
        hv.check_number_of_vfs_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=7)

    def test_check_number_of_vfs_by_pci_no_vfs_present(self, hv):
        output = "ls: cannot access '/sys/bus/pci/devices/0000:18:10.1/virtfn*': No such file or directory"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stderr=output, return_code=2)
        hv.check_number_of_vfs_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=0)

    def test_check_number_of_vfs_by_pci_incorrect(self, hv):
        output = dedent(
            """\
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn0 -> ../0000:af:02.0
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn1 -> ../0000:af:02.1
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn2 -> ../0000:af:02.2
            lrwxrwxrwx 1 root root 0 Jul 18 17:09 /sys/bus/pci/devices/0000:18:10.1/virtfn3 -> ../0000:af:02.3"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(
            args="", stdout=output, stderr="", return_code=0
        )
        with pytest.raises(VFExceptionKVM):
            hv.check_number_of_vfs_by_pci(pci_address=PCIAddress(0, 24, 16, 1), vfs_count=7)

    def test_set_trunk_add_action(self, hv):
        hv.set_trunk(pf_interface="eth1", action="add", vlan_id="200", vf_id=5)
        hv._conn.execute_command.assert_called_once_with(
            "echo add 200 > /sys/class/net/eth1/device/sriov/5/trunk",
            custom_exception=KVMHypervisorExecutionException,
            shell=True,
        )

    def test_set_trunk_rem_action(self, hv):
        hv.set_trunk(pf_interface="eth1", action="rem", vlan_id="200", vf_id=2)
        hv._conn.execute_command.assert_called_once_with(
            "echo rem 200 > /sys/class/net/eth1/device/sriov/2/trunk",
            custom_exception=KVMHypervisorExecutionException,
            shell=True,
        )

    def test_set_trunk_unsupported_action(self, hv):
        with pytest.raises(
            KVMHypervisorException,
            match="Unsupported action: not_supported_action, please use 'add' or 'rem'.",
        ):
            hv.set_trunk(
                pf_interface="eth1",
                action="not_supported_action",
                vlan_id="200",
                vf_id=7,
            )

    def test_get_trunk(self, hv):
        output = "200"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_trunk("eth1", vf_id=3) == "200"

    def test_set_tpid(self, hv):
        hv.set_tpid(interface="eth3", tpid="88a8")
        hv._conn.execute_command.assert_called_once_with(
            "echo 88a8 > /sys/class/net/eth3/device/sriov/tpid",
            custom_exception=KVMHypervisorExecutionException,
            shell=True,
        )

    def test_get_tpid(self, hv):
        output = "88a8"
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        assert hv.get_tpid("eth3") == "88a8"

    def test_attach_pci_controllers_single_device(self, hv, mocker):
        name = "vm_name"
        hv.shutdown_gracefully_vm = mocker.Mock()
        hv.get_vm_status = mocker.Mock(return_value={"State": "shut off"})
        hv.attach_pci_controllers(
            name=name,
            number_of_devices=1,
            first_index=1,
            first_chassis=0x10,
            first_port=0x11,
            first_bus=0x12,
            first_slot=0x03,
            first_func=0x01,
        )
        hv.virt_tool.attach_device.assert_called_once()

    def test_attach_pci_controllers_not_enough_pcis(self, hv, mocker):
        hv.shutdown_gracefully_vm = mocker.Mock()
        hv.get_vm_status = mocker.Mock(return_value={"State": "shut off"})
        with pytest.raises(
            KVMHypervisorException,
            match="Not enough free PCI devices. Cannot create expected number of PCI Controllers: expected: 5, "
            "created: 1",
        ):
            hv.attach_pci_controllers(
                name="vm_name",
                number_of_devices=5,
                first_index=1,
                first_chassis=0x10,
                first_port=0x11,
                first_bus=0x12,
                first_slot=0x1F,
                first_func=0x07,
            )

    def test_attach_pci_controllers_more_than_7(self, hv, mocker):
        hv.shutdown_gracefully_vm = mocker.Mock()
        hv.attach_device = mocker.Mock()
        hv.attach_pci_controllers(
            name="vm_name",
            number_of_devices=9,
            first_index=1,
            first_chassis=0x10,
            first_port=0x11,
            first_bus=0x12,
            first_slot=0x1E,
            first_func=0x06,
        )
        hv.virt_tool.start_vm.assert_called_once()

    def test_attach_pci_controllers_attach_failed(self, hv, mocker):
        hv.shutdown_gracefully_vm = mocker.Mock()
        hv.attach_device = mocker.create_autospec(
            hv.attach_device,
            side_effect=[
                None,
                None,
                None,
                KVMHypervisorExecutionException(
                    returncode=1,
                    cmd="",
                    output="",
                    stderr="Attempted double use of PCI Address 0000:01:01:11",
                ),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
        hv.attach_pci_controllers(
            name="vm_name",
            number_of_devices=9,
            first_index=1,
            first_chassis=0x10,
            first_port=0x11,
            first_bus=0x12,
            first_slot=0x1E,
            first_func=0x05,
        )
        hv.virt_tool.start_vm.assert_called_once()

    def test_is_vf_attached_true(self, mocker, hv):
        output = dedent(
            """\
        17:02.2 Ethernet controller: Intel Corporation Ethernet Controller XL710 for 40GbE QSFP+ (rev 02)
            Subsystem: Intel Corporation Ethernet Converged Network Adapter XL710-Q2
            Kernel driver in use: i40e
            Kernel modules: i40e
        18:10.0 Ethernet controller: Intel Corporation XL710/X710 Virtual Function (rev 02)
            Subsystem: Intel Corporation Device 0000
            Kernel modules: i40evf
        18:10.1 Ethernet controller: Intel Corporation XL710/X710 Virtual Function (rev 02)
            Subsystem: Intel Corporation Device 0000
            Kernel driver in use: vfio-pci
            Kernel modules: i40evf"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        hv.get_pci_address_for_vf = mocker.Mock(return_value=PCIAddress(0, 24, 16, 1))
        assert hv.is_vf_attached(interface="eth1", vf_id=0) is True

    def test_is_vf_attached_false(self, mocker, hv):
        output = dedent(
            """\
        18:10.1 Ethernet controller: Intel Corporation Ethernet Controller XL710 for 40GbE QSFP+ (rev 02)
            Subsystem: Intel Corporation Ethernet Converged Network Adapter XL710-Q2
            Kernel driver in use: i40e
            Kernel modules: i40e"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        hv.get_pci_address_for_vf = mocker.Mock(return_value=PCIAddress(0, 24, 16, 1))
        assert hv.is_vf_attached(interface="eth1", vf_id=0) is False

    def test_is_vf_attached_vf_pci_is_missing(self, mocker, hv):
        output = dedent(
            """\
        17:02.0 Ethernet controller: Intel Corporation Ethernet Controller XL710 for 40GbE QSFP+ (rev 02)
            Subsystem: Intel Corporation Ethernet Converged Network Adapter XL710-Q2
            Kernel driver in use: i40e
            Kernel modules: i40e"""
        )
        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)
        hv.get_pci_address_for_vf = mocker.Mock(return_value=PCIAddress(0, 24, 16, 1))
        with pytest.raises(
            KVMHypervisorException,
            match="VF PCI: 18:10.1 is missing in `lspci -k` output. Cannot check VF attaching state.",
        ):
            hv.is_vf_attached(interface="eth1", vf_id=0)

    def test_dump_xml_from_vm(self, hv):
        hv.virt_tool.dump_xml_from_vm.return_value = "<xml><tag1>content</tag1></xml>"
        assert isinstance(hv.dump_xml_from_vm("VM_name"), str)
        hv.virt_tool.dump_xml_from_vm.assert_called_once()

    def test_detach_interface_from_vm(self, hv):
        hv.virt_tool.detach_interface_from_vm.return_value = True
        hv.detach_interface_from_vm("VM_name", "FF:AA:BB:CC:DD:EE")
        hv.virt_tool.detach_interface_from_vm.assert_called_once()

    def test_list_vms(self, hv):
        hv.virt_tool.list_vms.return_value = [{"Name": "VM1"}]
        assert isinstance(hv.list_vms(), list)
        hv.virt_tool.list_vms.assert_called_once()

    def test_list_vms_all(self, hv):
        hv.virt_tool.list_vms.return_value = [{"Name": "VM1"}]
        assert isinstance(hv.list_vms(True), list)
        hv.virt_tool.list_vms.assert_called_once()

    def test_get_mac_for_mng_vm_interface(self, hv):
        hv.virt_tool.get_mac_for_mng_vm_interface.return_value = "52:54:00:a3:ac:dc"
        assert isinstance(hv.get_mac_for_mng_vm_interface("VM_name"), str)
        hv.virt_tool.get_mac_for_mng_vm_interface.assert_called_once()

    def test_get_mng_ip_for_vm(self, hv):
        hv.virt_tool.get_mng_ip_for_vm.return_value = "172.20.0.1"
        hv.get_mng_ip_for_vm("mac", "vm_id")
        hv.virt_tool.get_mng_ip_for_vm.assert_called_once()

    def test_get_guest_mng_ip(self, hv, mocker):
        mocker.patch("mfd_kvm.KVMHypervisor.get_mac_for_mng_vm_interface", return_value="mac")
        mocker.patch("mfd_kvm.KVMHypervisor.get_mng_ip_for_vm", return_value="172.20.0.1")
        assert hv.get_guest_mng_ip("vm_id") == "172.20.0.1"

    def test_get_guest_mng_ip_raise(self, hv, mocker):
        hv.get_mac_for_mng_vm_interface = mocker.MagicMock(side_effect=VirshException(1, "Error"))
        with pytest.raises(KVMHypervisorException, match="Cannot find MAC address for VM: vm_id"):
            hv.get_guest_mng_ip("vm_id")

    def test_set_vcpus(self, hv):
        hv.set_vcpus("VM_Name", 16)
        hv.virt_tool.set_vcpus.assert_called_once()

    def test_set_vcpus_max_limit(self, hv):
        hv.set_vcpus_max_limit("VM_Name", 16)
        hv.virt_tool.set_vcpus_max_limit.assert_called_once()

    def test_create_vm_network(self, hv):
        hv.virt_tool.create_vm_network.return_value = True
        assert isinstance(hv.create_vm_network("<xml><tag1>content</tag1></xml>"), bool)
        hv.virt_tool.create_vm_network.assert_called_once()

    def test_destroy_vm_network(self, hv):
        hv.virt_tool.destroy_vm_network.return_value = True
        assert isinstance(hv.destroy_vm_network("<xml><tag1>content</tag1></xml>"), bool)
        hv.virt_tool.destroy_vm_network.assert_called_once()

    def test_attach_tap_interface_to_vm(self, hv):
        hv.virt_tool.attach_tap_interface_to_vm.return_value = True
        assert isinstance(hv.attach_tap_interface_to_vm("VM_Name", "NetConfig"), bool)
        hv.virt_tool.attach_tap_interface_to_vm.assert_called_once()

    def test_attach_interface(self, hv, mocker, caplog):
        mocker.patch("mfd_kvm.KVMHypervisor.prepare_vf_xml", return_value="Path_to_xml")
        hv.virt_tool.get_vm_status.return_value = {"State": "Some_State"}
        caplog.set_level(log_levels.MODULE_DEBUG)
        hv.attach_interface("VM_Name", PCIAddress(4000, 3, 22, 11))
        hv.virt_tool.attach_device.assert_called_once()

    def test_detach_interface(self, hv, mocker, caplog):
        mocker.patch("mfd_kvm.KVMHypervisor.prepare_vf_xml", return_value="Path_to_xml")
        hv.virt_tool.get_vm_status.return_value = {"State": "Some_State"}
        caplog.set_level(log_levels.MODULE_DEBUG)
        hv.detach_interface("VM_Name", PCIAddress(4000, 3, 22, 11))
        hv.virt_tool.detach_device.assert_called_once()

    def test_get_vfs_details_for_interface_pass(self, hv):
        interface_name = "eth1"

        expected_command = f"ip link show dev {interface_name}"
        output = dedent(
            """
        3: eth1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
        link/ether aa:bb:cc:de:ed:be brd ff:ff:ff:ff:ff:ff
        vf 0     link/ether 00:00:00:00:00:00 brd ff:ff:ff:ff:ff:ff, spoof checking on, link-state auto, trust off
        vf 1     link/ether 00:00:00:00:00:00 brd ff:ff:ff:ff:ff:ff, spoof checking on, link-state auto, trust off
        vf 9     link/ether 00:00:00:00:00:00 brd ff:ff:ff:ff:ff:ff, spoof checking on, link-state auto, trust off
        """
        )

        hv._conn.execute_command.return_value = ConnectionCompletedProcess(args="", stdout=output, return_code=0)

        expected_details = [
            VFDetail(
                id=0,
                mac_address=MACAddress("00:00:00:00:00:00"),
                spoofchk=True,
                trust=False,
            ),
            VFDetail(
                id=1,
                mac_address=MACAddress("00:00:00:00:00:00"),
                spoofchk=True,
                trust=False,
            ),
            VFDetail(
                id=9,
                mac_address=MACAddress("00:00:00:00:00:00"),
                spoofchk=True,
                trust=False,
            ),
        ]

        assert hv.get_vfs_details_from_interface(interface_name=interface_name) == expected_details
        hv._conn.execute_command.assert_called_with(
            command=expected_command, custom_exception=KVMHypervisorExecutionException
        )

    def test_get_vfs_details_for_interface_error(self, hv):
        hv._conn.execute_command.side_effect = KVMHypervisorExecutionException(1, "", "", "")
        with pytest.raises(KVMHypervisorExecutionException):
            hv.get_vfs_details_from_interface(interface_name="foo")

    def test_get_vf_id_from_mac_address_pass(self, hv, mocker):
        vf_details = [
            VFDetail(
                id=0,
                mac_address=MACAddress("00:00:00:00:00:00"),
                spoofchk=True,
                trust=False,
            ),
            VFDetail(
                id=1,
                mac_address=MACAddress("00:00:00:00:00:01"),
                spoofchk=True,
                trust=False,
            ),
            VFDetail(
                id=9,
                mac_address=MACAddress("00:00:00:00:00:02"),
                spoofchk=True,
                trust=False,
            ),
        ]

        hv.get_vfs_details_from_interface = mocker.Mock(return_value=vf_details)
        assert hv.get_vf_id_from_mac_address(interface_name="foo", mac_address=MACAddress("00:00:00:00:00:01")) == 1
