# Hardware

8× NVIDIA CMP 170HX (GA100 silicon, sm_80), VRAM-unlocked to 64 GB each, VBIOS
`92.00.6D.00.0A` uniformly across all 8 — see [BIOS.md](BIOS.md).

## Why PLX switches

The 170HX is a mining card: no display output, and cheap ones ship on boards that
only expose a **PCIe Gen2 x4** edge connector — nowhere near enough host slots or
lanes to wire 8 of them straight to a workstation motherboard. The setup here fans
each host x16 slot out through a **PLX PEX8749** switch (48-lane, 18-port Gen3 switch
with DMA) to multiple downstream x4 links, one per card.

```
lspci -d 10b5:        # list PLX switches (vendor id 10b5)
lspci | grep -i nvidia
```

Two PEX8749 switches are in use, each fanning out to 4 GPUs:

- root port `86:00.0` / switch fan-out at `87:08.0`–`87:13.0` → GPUs at `89:00.0`,
  `8b:00.0`, `8d:00.0`, `8f:00.0`
- root port `af:00.0` / switch fan-out at `b0:08.0`–`b0:13.0` → GPUs at `b2:00.0`,
  `b4:00.0`, `b6:00.0`, `b8:00.0`

## Risers and cabling

**C-Payne** PCIe riser/adapter boards, connected with **SFF-8654** cables — each card
gets its own **x4 PCIe Gen2** run from the switch to the card's edge connector,
matching the card's native x4 link (no benefit to wiring more lanes than the card can
use).

## Known pitfall: `PWRBRK#` / edge pin B30

Some boards assert the `PWRBRK#` sideband signal (edge pin B30) and force the card
into a permanent hardware power brake — full clocks and power budget never apply.
Check with:

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
```

`Active` on a card that isn't thermally or power limited means the platform is
asserting it. Fix: Kapton tape over B30 on the card edge connector, or a riser that
doesn't route B30.
