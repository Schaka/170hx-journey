# VBIOS

All 8 cards run VBIOS `92.00.6D.00.0A`, `clocks.max.memory` 1728 MHz on every card —
uniform across the whole box.

```bash
nvidia-smi --query-gpu=index,vbios_version,clocks.max.memory,pci.bus_id --format=csv
```

Cards that shipped with the older `92.00.67.00.01` VBIOS were capped at 1458 MHz max
memory clock — an ~18.5% lower memory-bandwidth ceiling than `6D`. Flashed to `6D`
using `nvflash` and the `92.00.6D.00.0A` ROM (same subsystem ID `10DE:1585`, device ID
`0x20C2` as the cards' native VBIOS, so a straight flash rather than a cross-flash).

## Flashing procedure used

1. Stop every container/process using the GPUs (`nvidia-smi --query-compute-apps`
   must come back empty).
2. Unload the nvidia driver modules (`nvidia_uvm`, `nvidia_drm`, `nvidia_modeset`,
   `nvidia`) so `nvflash` can `mmap()` the card's BAR directly — it fails with an
   `mmap(): /dev/mem` error otherwise.
3. `nvflash -i<N> --save backup.rom` per target card before touching anything.
4. `nvflash -i<N> <target>.rom`, confirming the version/ID match it prints before
   accepting.
5. Reboot — the new VBIOS doesn't take effect until then.

`nvflash` reads its yes/no confirmation from the controlling TTY directly, not stdin —
piping input to it over a non-interactive SSH command does nothing; it needs a real
pty (`ssh -tt`, or a pty-driving wrapper) to answer the prompt.
