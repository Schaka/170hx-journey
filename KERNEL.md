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

[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) makes this setup work. The
kernel patches for the bridge-window bug came from that project. The full driver patch
set for real BAR1 P2P came from that project too, not the mailbox path. This project is
the reason 64 GB BAR1 behind a PLX switch works on this hardware. This project is also
the reason P2P between these cards works. This repository also refers to
[cachenetics/170tune](https://github.com/cachenetics/170tune) for general 170HX tuning
context.

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
pci=realloc pci=hpmmioprefsize=2T
pci=disable_acs_redir=0000:85:00.0
pci=disable_acs_redir=0000:86:00.0
pci=disable_acs_redir=0000:87:00.0
pci=disable_acs_redir=0000:87:11.0
pci=disable_acs_redir=0000:87:13.0
pci=disable_acs_redir=0000:ae:00.0
pci=disable_acs_redir=0000:af:00.0
pci=disable_acs_redir=0000:b0:00.0
pci=disable_acs_redir=0000:b0:11.0
pci=disable_acs_redir=0000:b0:13.0
intel_iommu=on iommu=pt
```

Name every bridge along each GPU-to-GPU path in `disable_acs_redir`. Name the outer
root ports. Also name the PLX switch's own internal ports, the `87:xx.x` and `b0:xx.x`
entries above. If you skip an internal port, ACS redirection still blocks P2P traffic
between cards on the same switch. The `pci=hpmmioprefsize=2T` argument gives enough
prefetchable MMIO space for all eight cards' 64 GB BAR1 windows, plus headroom.

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
