# Kernel and driver

This server runs a custom-built kernel and a custom-built NVIDIA driver module. Both
are required. A stock Fedora kernel and a stock NVIDIA driver cannot give 64 GB BAR1 or
working GPU-to-GPU P2P on this hardware.

## Why a stock kernel cannot do this job

Two problems block a stock kernel here. Both problems are specific to running these
cards behind PLX switches.

1. Bridge window sizing. A Linux kernel bug in `pbus_size_mem()`
   (`drivers/pci/setup-bus.c`) computes a PCI bridge window with
   `size += max(r_size, align)` instead of `size += ALIGN(r_size, align)`. Every
   downstream port on the PLX switches needs a 64 GB BAR1 window and a 32 MB BAR3
   window, per card. When a child BAR's size is not an exact multiple of its own alignment, the unpatched
   formula under-sizes the parent bridge window. This card triggers that exact
   condition. Without the fix, a BAR1 resize behind a PLX switch cannot
   succeed.
2. The mailbox P2P mechanism does not work on the CMP 170HX. The driver's default
   `P2P_CONNECTIVITY_PCIE_PROPRIETARY` mailbox mechanism reports success. Each GSP RPC
   call returns `NV_OK`. But the mechanism moves zero bytes. It gives no error and no
   Xid fault. It fails silently. Real P2P on this card needs the BAR1 P2P mechanism
   instead. BAR1 P2P rewrites peer page table entries from `GMMU_APERTURE_PEER` to
   `SYS_COH` or `SYS_NONCOH`. Each entry then points directly at the peer's BAR1 bus
   address. The stock driver does not contain this mechanism.

## Thanks

- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) (v0.3):
  the base driver build this kernel runs, including the `610.57.04` source lineage
  and the memory-geometry, BAR1-size, and PCIe Gen2 unlock patches.
- [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker): the kernel patch for
  the bridge-window bug above, and the full driver patch set for real BAR1 P2P
  instead of the dead mailbox path. This project is the reason 64 GB BAR1 behind a
  PLX switch works on this hardware, and the reason P2P between these cards works.
- [cachenetics/170tune](https://github.com/cachenetics/170tune): general 170HX
  tuning reference.

## Kernel

The custom-built kernel is `7.1.10-cmp`. It is based on the stock Fedora 7.1.10
source. It has two patches from `bayley/cmpunlocker`'s `kernel-patches/` directory:

- A bridge-window sizing fix for child-BAR alignment. This fix corrects the
  `pbus_size_mem()` bug above.
- An early ReBAR quirk for the CMP 170HX.

Set this kernel as the persistent default boot kernel. Use `grubby --set-default` for
this, not a one-time `grub2-reboot` override. A one-time override gets consumed by the
next reboot. After that, the server falls back to a stock kernel with no BAR1 and no
P2P. This failure looks like a hang at NCCL init, in any container that starts on the
stock kernel.

### Kernel command line

```
ro rootflags=subvol=root rhgb quiet
intel_iommu=off iommu=off
pci=realloc pci=hpmmioprefsize=2T
pci=disable_acs_redir=0000:85:00.0
pci=disable_acs_redir=0000:87:09.0
pci=disable_acs_redir=0000:87:0b.0
pci=disable_acs_redir=0000:87:11.0
pci=disable_acs_redir=0000:87:13.0
pci=disable_acs_redir=0000:ae:00.0
pci=disable_acs_redir=0000:b0:09.0
pci=disable_acs_redir=0000:b0:0b.0
pci=disable_acs_redir=0000:b0:11.0
pci=disable_acs_redir=0000:b0:13.0
rd.driver.blacklist=nouveau rd.driver.blacklist=nova-core
```

The `disable_acs_redir` list names the two GPU-side root ports (`85:00.0`, `ae:00.0`)
and all eight PLX switch downstream ports, one per card. The `pci=hpmmioprefsize=2T`
argument gives enough prefetchable MMIO space for all eight cards' 64 GB BAR1 windows,
plus headroom. `rd.driver.blacklist` keeps the kernel's own `nouveau`/`nova-core`
drivers from claiming the cards before the patched NVIDIA driver does.

`pci=disable_acs_redir` only changes the kernel's own P2P-DMA completion routing. It
does **not** clear the ACS control register bit in PCIe config space on the affected
bridges. GPU-to-GPU P2P on this hardware uses the BAR1 rewrite mechanism (see above).
That mechanism routes through MMIO, not a hardware DMA engine. It works even with ACS
still enabled at the hardware level.

A DMA engine that bus-masters directly between two peers does need the ACS bit
actually cleared in hardware. An NVMe controller doing GPUDirect Storage is one
example. `disable_acs_redir` alone does not clear that bit. See
[the GDS section below](#gpudirect-storage-status) for how this box clears it, and
[the runbook above](#finding-and-changing-kernel-arguments-on-this-box) for the
general method.

### Finding and changing kernel arguments on this box

Use this method whenever a new piece of hardware or a new driver feature needs a
kernel argument this repo does not already document.

1. Read the running kernel's actual command line, not what a doc claims it is:
   ```bash
   cat /proc/cmdline
   grubby --info=/boot/vmlinuz-7.1.10-cmp | grep args
   ```
   These two must agree. If they do not, the persistent boot entry and the kernel
   that is actually running are out of sync.
2. Map the PCIe topology for the device you care about:
   ```bash
   lspci -tv                                   # tree view, bridges and depth
   lspci -d 10de:                               # NVIDIA GPUs
   lspci -d 10b5:                               # PLX switches
   ```
   Every bridge between your two endpoints (the GPU and the peer device) is a
   candidate for `disable_acs_redir`.
3. Check whether ACS is actually enabled on a given bridge, in hardware, not just in
   the kernel's redirect logic:
   ```bash
   sudo lspci -vvv -s <bridge BDF> | grep -A2 'Capabilities:.*Access Control Services'
   ```
   This prints a block like:
   ```
   Capabilities: [f24 v1] Access Control Services
           ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl+ DirectTrans+
           ACSCtl: SrcValid+ TransBlk- ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-
   ```
   The number in `[f24 v1]` is the capability's byte offset into PCI config space.
   `ACSCtl` is the live control register. Any bit shown with `+` is active. `ACSCap`
   only lists which bits this bridge supports, not which ones are on.
4. If a bridge on your path shows `ACSCtl` bits set, clear ACS in hardware with
   `setpci`. The control register sits 6 bytes after the capability offset from step
   3: a 4-byte capability header, then a 2-byte `ACSCap` field. For a capability at
   `[f24]`, the control register is at `0xf2a`:
   ```bash
   sudo setpci -s <bridge BDF> 0xf2a.w=0000
   ```
   Repeat the `lspci` command from step 3 to check it took. Every `ACSCtl` bit must
   now read `-`. This write is live and immediate. It needs no reboot to test.

   It does not survive a reboot on its own, because PCI config space resets to
   firmware defaults on every power cycle. Turn it into a systemd oneshot service,
   ordered before whatever driver depends on it. See
   [`clear-acs.service`](#example-clear-acsservice) below. Skip that step and it
   silently reverts on the next boot.
5. Change kernel *arguments* (not a live hardware register) with `grubby`. Pass the
   complete desired list in one call. Two separate `grubby --args=` calls do not
   merge. The second call replaces the whole token group from the first, and
   silently drops earlier entries:
   ```bash
   sudo grubby --update-kernel=/boot/vmlinuz-7.1.10-cmp --args='<the full args string>'
   ```
6. Check the change landed, before you reboot:
   ```bash
   grubby --info=/boot/vmlinuz-7.1.10-cmp | grep args
   ```
7. Reboot, then re-run the checks in
   [How to check that it worked](#how-to-check-that-it-worked). Check that P2P did not
   regress before you move on to testing whatever the new argument was for.

### Example: `clear-acs.service`

A systemd oneshot unit that reapplies an ACS clear on every boot, before the driver
that needs it loads:

```ini
# /etc/systemd/system/clear-acs.service
[Unit]
Description=Clear PCIe ACS on GPU/NVMe switch ports (GPUDirect Storage)
DefaultDependencies=no
Before=sysinit.target
ConditionPathExists=/usr/local/sbin/clear-acs.sh

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/clear-acs.sh
RemainAfterExit=yes

[Install]
WantedBy=sysinit.target
```

```bash
# /usr/local/sbin/clear-acs.sh
#!/bin/bash
set -euo pipefail
PORTS=(87:09.0 87:0b.0 87:11.0 87:13.0 b0:09.0 b0:0b.0 b0:11.0 b0:13.0)
for bdf in "${PORTS[@]}"; do
    setpci -s "${bdf}" 0xf2a.w=0000
done
```

`Before=sysinit.target` with `DefaultDependencies=no` runs this early enough to beat
`nvidia_fs`'s module load. Enable it with
`systemctl enable --now clear-acs.service`.

## GPUDirect Storage status

GDS works. [`gdsio`](#checking-gds-with-gdsio) reports `XferType: GPUD` on both read
and write against the NVMe. That is proof of a real GPUDirect transfer, and it
survives a cold reboot.

The install uses [cachenetics/gds-nvme-patch](https://github.com/cachenetics/gds-nvme-patch):
a patched `nvme.ko` built for kernel `7.1.10-cmp`, plus `nvidia-fs` 2.29.4.
`nvidia_fs` exposes its state at `/proc/driver/nvidia-fs/modules`, and
`gdscheck.py -p` reports `nvfs, compat` for the NVMe path. Both of these describe
`nvidia_fs`'s own static capability negotiation, not the runtime path an actual
transfer takes. Trust `gdsio`'s `XferType` field over either of them.

What made it work: `gdscheck.py -p` named the real blocker directly, in its
`PLATFORM INFO` section: `Found ACS enabled for switch 0000:87:09.0` (and seven more,
one per PLX switch downstream port between the GPUs and the NVMe). `disable_acs_redir`
in the kernel command line does not clear that bit. See the note above. Clearing
`ACSCtl` directly with `setpci` on all eight ports removed every one of those
warnings from `gdscheck`'s report, and made `gdsio` show `GPUD`.
[`clear-acs.service`](#example-clear-acsservice) above makes that persist across a
reboot.

### Checking GDS with `gdsio`

```bash
mkdir -p ~/gdstest
sudo /usr/local/cuda-13.3/gds/tools/gdsio -D ~/gdstest -d 0 -w 1 -s 32M -i 1M -x 0 -I 1 -V
rm -rf ~/gdstest
```

Look for `XferType: GPUD` in the output, on both the write and the read line.
`XferType: CPUD` or a similar non-`GPUD` value means the transfer fell back to a
CPU-staged copy. `gdscheck.py -p` can report `Supported` and this can still happen.

## Driver module

The driver module build starts from a fresh `NVIDIA/open-gpu-kernel-modules` checkout,
tag `610.57.04`. The build applies the full `bayley/cmpunlocker` driver patch set: 12
patches, plus three new `cmpunlock` source files wired into `srcs.mk`. The build
targets the `7.1.10-cmp` kernel headers. The module installs under
`/lib/modules/7.1.10-cmp/updates/cmpunlocker/`.

File `/etc/depmod.d/cmpunlocker.conf` forces this path to win module resolution over
any stock driver a package manager installs later, at `extra/` or `updates/dkms/`:

```
override nvidia * updates/cmpunlocker
override nvidia-modeset * updates/cmpunlocker
override nvidia-uvm * updates/cmpunlocker
override nvidia-drm * updates/cmpunlocker
override nvidia-peermem * updates/cmpunlocker
```

### Upgrading the driver version

`bayley/cmpunlocker`'s own `driver/VERSION` file pins to `610.43.0x`. It does not
track every upstream NVIDIA release. The userspace NVIDIA stack can move to a newer
version on its own, through a `dnf` dependency chain (see the gotcha below). When that
happens, port the patch set by hand.

This worked cleanly at least once, from `610.43.03` to `610.57.04`. The port needed
only line-offset adjustments and one small fuzz match. No hunk was rejected.

1. Fetch a fresh source tree for the target version:
   ```bash
   curl -L --fail -o ogkm-<VERSION>.tar.gz \
     https://github.com/NVIDIA/open-gpu-kernel-modules/archive/refs/tags/<VERSION>.tar.gz
   tar -xzf ogkm-<VERSION>.tar.gz
   ```
2. Apply the 12 `bayley/cmpunlocker` patches in order, against that fresh tree:
   `0001-gsp-hooks`, `0002-tu102-hook`, `0003-osinit-late-pma-hook`,
   `0004-memmgr-quirks`, `0005-bar0-pramin-clamp`, `0006-persistent-sw-state`,
   `0007-p2p-caps-hook`, `0008-scrub-timeout`, `0009-bar1-resize-unlock`,
   `0011-p2p-bar1`, `0013-skip-mailbox-peer-preinit`, `0015-bar1p2p-readcap-override`.
   Run `patch -p1 --dry-run` first for every patch, read each report, and fix a reject
   by hand before applying for real. Do not skip a patch that fails silently.
3. Copy the three `cmpunlock` source files from a prior working tree into the same
   paths in the new tree: `src/nvidia/src/kernel/gpu/cmpunlock/cmpunlock.c`,
   `src/nvidia/inc/kernel/gpu/cmpunlock/cmpunlock.h`, and
   `src/nvidia/inc/kernel/gpu/cmpunlock/cmpunlock_config.h`. Append
   `SRCS += src/kernel/gpu/cmpunlock/cmpunlock.c` to `src/nvidia/srcs.mk`.
4. Build against the running patched kernel's headers, and check the `cmpUnlock*`
   symbols appear in the build log:
   ```bash
   make -j$(nproc) modules SYSSRC=/lib/modules/7.1.10-cmp/build
   ```
5. Before you touch the live system, back up the current, working
   `/lib/modules/7.1.10-cmp/updates/cmpunlocker/*.ko` files to a dated directory under
   `/root/`. Stop every container using the GPUs first.
6. Copy the five new `.ko` files into `/lib/modules/7.1.10-cmp/updates/cmpunlocker/`,
   write the new version string to `driver_version` in that directory, then run
   `depmod -a 7.1.10-cmp` and `dracut -f --kver 7.1.10-cmp`.
7. Check resolution before you reboot: `modinfo nvidia` must report the new
   `filename` under `updates/cmpunlocker/` and the new `version`.
8. Reboot, then run the checks in [How to check that it worked](#how-to-check-that-it-worked)
   below. If P2P reads `NS` on any pair, or `nvidia-smi` fails, do not try to fix it
   live. Copy the backed-up `.ko` files from step 5 back into place, `depmod` and
   `dracut` again, and reboot back to the known-good version.

### Gotcha: a package install can silently change the default boot kernel

Installing an unrelated package (`nvidia-fs-dkms`, in one case) can pull in a new
stock kernel as a dependency. The RPM kernel scriptlets that run during that install
can silently reset the persistent default boot kernel to the new one. Check the
default did not change, right after any `dnf`/`rpm` transaction that touches a kernel
package:

```bash
grubby --default-kernel   # must read /boot/vmlinuz-7.1.10-cmp
```

If it changed, restore it before the next reboot: `grubby --set-default=/boot/vmlinuz-7.1.10-cmp`.

### Required registry dwords

File `/etc/modprobe.d/nvidia-p2p.conf` contains this line:

```
options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1"
```

Both parameters are required together. `RMPcieP2PType=1` removes the mailbox's roughly
512 KB footprint inside BAR1. Without that parameter, static BAR1 allocation fails,
because the mailbox pushes the total past 64 GB.

## How to check that it worked

```bash
nvidia-smi --query-gpu=index,vbios_version,clocks.max.memory,pci.bus_id --format=csv
nvidia-smi topo -p2p p       # every pair must read OK, not NS
uname -r                     # must read 7.1.10-cmp
```
