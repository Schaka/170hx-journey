# Kernel and driver

This box runs a **custom-built kernel and a custom-built NVIDIA driver module**. Both
are required — a stock Fedora kernel and stock NVIDIA driver cannot deliver 64 GB BAR1
or working GPU-to-GPU P2P on this hardware, full stop.

## Why a stock kernel can't do this

Two independent problems, both specific to running these cards behind PLX switches:

1. **Bridge window sizing.** A Linux kernel bug in `pbus_size_mem()`
   (`drivers/pci/setup-bus.c`) computes a PCI bridge window as
   `size += max(r_size, align)` instead of `size += ALIGN(r_size, align)`. Every
   downstream port on the PLX switches here needs a 64 GB BAR1 + 32 MB BAR3 window per
   card, and the unpatched arithmetic silently under-sizes the parent bridge window
   whenever a child BAR's size isn't an exact multiple of its own alignment — exactly
   this case. Without the fix, BAR1 resize behind a PLX switch cannot succeed at all.
2. **Mailbox-style P2P is a dead end on the CMP 170HX.** The driver's default
   `P2P_CONNECTIVITY_PCIE_PROPRIETARY` mailbox mechanism reports success and returns
   `NV_OK` from every GSP RPC, but moves zero bytes — no error, no Xid, just silently
   broken. Real P2P on this card requires the **BAR1 P2P** mechanism instead: rewriting
   peer page table entries from `GMMU_APERTURE_PEER` to `SYS_COH`/`SYS_NONCOH`,
   pointing directly at the peer's BAR1 bus address. This isn't in the stock driver at
   all.

## Thanks

[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) is the reason any of this
works: the kernel patches for the bridge-window bug, and the full driver patch set for
real BAR1 P2P (not the mailbox path), came from that project. Without it, both 64 GB
BAR1 behind a PLX switch and working P2P between these cards would have been
unachievable on this hardware. Also referencing
[cachenetics/170tune](https://github.com/cachenetics/170tune) for general 170HX tuning
context.

## Kernel

Custom-built kernel `7.1.10-cmp`, based on the stock Fedora 7.1.10 source, with two
patches from `bayley/cmpunlocker`'s `kernel-patches/` applied:

- bridge-window sizing fix for child-BAR alignment (the `pbus_size_mem()` bug above)
- an early ReBAR quirk for the CMP 170HX

Set as the **persistent** default boot kernel (`grubby --set-default`), not a
one-time `grub2-reboot` override — a one-time override gets silently consumed by the
next reboot and the box falls back to a stock kernel with no BAR1/P2P, which looks
like a hang at NCCL init in any container that starts on it.

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

`disable_acs_redir` has to name **every** bridge along each GPU-to-GPU path, not just
the outer root ports — that includes the PLX switch's own internal ports (the
`87:xx.x` / `b0:xx.x` entries above), or ACS redirection still blocks P2P traffic
between cards hanging off the same switch. `pci=hpmmioprefsize=2T` gives enough
prefetchable MMIO space for all 8 cards' 64 GB BAR1s plus headroom.

## Driver module

Built from a fresh `NVIDIA/open-gpu-kernel-modules` (`610.43.03`) checkout with the
full `bayley/cmpunlocker` driver patch set applied (12 patches plus the 3 new
`cmpunlock` source files, wired into `srcs.mk`), built against the `7.1.10-cmp`
kernel headers, installed under
`/lib/modules/7.1.10-cmp/updates/cmpunlocker/`.

### Required registry dwords

`/etc/modprobe.d/nvidia-p2p.conf`:

```
options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1"
```

Both params are required together — `RMPcieP2PType=1` removes the mailbox's ~512 KB
in-BAR1 footprint, and without it static BAR1 allocation fails because the mailbox
pushes the total just over 64 GB.

## Verifying it worked

```bash
nvidia-smi --query-gpu=index,vbios_version,clocks.max.memory,pci.bus_id --format=csv
nvidia-smi topo -p2p p       # every pair should read OK, not NS
uname -r                     # should be 7.1.10-cmp
```
