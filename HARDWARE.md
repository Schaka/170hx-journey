# Hardware

The server has eight NVIDIA CMP 170HX cards. Each card uses GA100 silicon (sm_80).
Each card has 64 GB of unlocked VRAM. Every card runs VBIOS `92.00.6D.00.0A`. See
[BIOS.md](BIOS.md) for the VBIOS record.

## Why PLX switches

The CMP 170HX is a mining card. It has no display output. Cheap cards ship on boards
that expose only a PCIe Gen2 x4 edge connector. A workstation motherboard does not
have enough slots or lanes to connect eight cards directly. This setup connects each
host x16 slot through a PLX PEX8749 switch. The switch is a 48-lane, 18-port Gen3
switch with DMA. Each switch fans out to several downstream x4 links, one per card.

Run these commands to list the switches and cards:

```
lspci -d 10b5:        # list PLX switches (vendor id 10b5)
lspci | grep -i nvidia
```

The server uses two PEX8749 switches. Each switch connects to four GPUs.

- Root port `86:00.0`. Switch fan-out at `87:08.0` to `87:13.0`. Connects to GPUs at
  `89:00.0`, `8b:00.0`, `8d:00.0`, and `8f:00.0`.
- Root port `af:00.0`. Switch fan-out at `b0:08.0` to `b0:13.0`. Connects to GPUs at
  `b2:00.0`, `b4:00.0`, `b6:00.0`, and `b8:00.0`.

## Risers and cables

The setup uses C-Payne PCIe riser and adapter boards. The boards connect through
SFF-8654 cables. Each card gets its own PCIe Gen2 x4 connection from the switch to the
card's edge connector. This matches the card's native x4 link. More lanes give no
benefit. The card cannot use them.

## Known problem: PWRBRK# on edge pin B30

Some boards assert the `PWRBRK#` sideband signal on edge pin B30. This signal forces
the card into a permanent hardware power brake. The card never reaches its full clocks
or power budget under this condition.

Run this command to check for the condition:

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
```

If the output reads `Active` on a card that is not thermally limited or power limited,
the platform asserts `PWRBRK#`. To fix this, place Kapton tape over pin B30 on the
card edge connector. You can also use a riser that does not route B30.
