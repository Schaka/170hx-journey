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

## Storage

The box has two RAID 0 arrays for model weights. RAID 0 stripes data across
drives for speed and gives no redundancy.

`/dev/md0` mounts at `/models`. It stripes four NVMe drives of 512 GB each and
gives 1.9 TiB. This array holds the models that the box serves every day.

`/dev/md1` mounts at `/backup-models`. It stripes two Intel SSDSC2BB012T4 SATA
SSDs of 1.2 TB each and gives 2.2 TiB. This array holds the models that the box
keeps but rarely serves.

Both arrays use a 512 KB chunk and an ext4 filesystem. `/etc/fstab` mounts both
by filesystem UUID with the `nofail` option. A missing array will not stop the
boot. `/etc/mdadm.conf` names `/dev/md1` by array UUID.

## Power limit and clock tuning

Each card ships with a 250 W limit and a 300 W hardware maximum. This host
runs every card at a 175 W limit, an undervolt, and a 1470 MHz clock ceiling.
Under that profile a card draws 38 to 96 W while it serves, so the limit is a
guard rail and not a brake.

### The tune

[cachenetics/170tune](https://github.com/cachenetics/170tune) writes the clock
and voltage registers through GPU BAR0. The kernel command line must carry
`iomem=relaxed` for that mapping to work. See
[KERNEL.md](KERNEL.md#kernel-command-line).

The tool needs `nvcc` to build its bit-exact compute check. This host has CUDA
13.3 from the `cuda-fedora44-x86_64` repository. That `nvcc` refuses a host
compiler above GCC 15, so the build needs `gcc15` and `gcc15-c++`, and this
environment variable:

```
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-15"
```

### Per-card offsets

The cards are not one bin. Cards 1 and 3 report a maximum SM clock of 1785
MHz. The other six report 1890 MHz. The two slower cards need a smaller
voltage offset, and 170tune keys every profile to a card serial.

| GPU | serial | offset | ceiling |
|---|---|---|---|
| 0 | 1322421042439 | +200 | 1470 MHz |
| 1 | 1322621048774 | +100 | 1470 MHz |
| 2 | 1322421043344 | +200 | 1470 MHz |
| 3 | 1322821045481 | +100 | 1470 MHz |
| 4 | 1322321055604 | +200 | 1470 MHz |
| 5 | 1322421000145 | +200 | 1470 MHz |
| 6 | 1322821057768 | +200 | 1470 MHz |
| 7 | 1322321010343 | +200 | 1470 MHz |

Cards 1 and 3 hold `+200` in quarantine. `170tune persist save` will refuse
that point on those two serials.

### The synthetic gate is not enough

`170tune gate` soaks a card, writes and reads back almost all of its 64 GB,
and runs about 59,000 bit-exact matrix multiplies. Every card passed that gate
at `+250`, and the real server still died. Card 3 threw `Xid 13 Illegal
Instruction Encoding` seconds into the first vLLM request. Card 1 threw the
same fault at `+200` after four clean benchmark rounds.

Both faults landed on `TPC 6, SM 1`. Only the real workload found them, so
qualify an offset with the engine that will serve, over many rounds. The
shipped offsets above survive 23 rounds of 8 concurrent streams plus 5
long-context needle checks, with no Xid.

Card 6 sits in a well cooled slot and peaks at 54 C HBM under a full-VRAM
sweep. It cannot reach the 60 C that a hot gate wants, so its receipt is cold.

### Boot path

Two units apply this at boot, in order:

- `170tune-persist.service` applies each card's own offset and ceiling. It
  reads `/var/lib/170tune/persist/<serial>.conf`.
- `gpu-power-limit.service` runs
  [`scripts/gpu-power-limit.sh`](scripts/gpu-power-limit.sh) at 175 W.

The order matters. 170tune raises the limit to 300 W as headroom, so the cap
unit declares `After=170tune-persist.service`.

The box always boots on stock values first. To recover from a bad profile,
run `systemctl mask 170tune-persist.service` over SSH and reboot.

### Thermals and faults

Under load the cards reach 52 to 70 C on the core and 58 to 73 C on the HBM.
The limits are 85 C and 95 C.

These cards have no HBM ECC and no row remapping. `nvidia-smi -q -d ECC` and
`--query-remapped-rows` both return `[N/A]`. A bad memory cell is permanent,
silent, and looks like a software problem. Run `dmesg | grep -c Xid` before
every launch.
