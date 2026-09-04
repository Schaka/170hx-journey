# VBIOS

All eight cards run VBIOS `92.00.6D.00.0A`. The `clocks.max.memory` value reads
1728 MHz on every card. The value is the same across the whole server.

```bash
nvidia-smi --query-gpu=index,vbios_version,clocks.max.memory,pci.bus_id --format=csv
```

Cards that shipped with the older `92.00.67.00.01` VBIOS had a max memory clock of
1458 MHz. That clock is a memory-bandwidth ceiling roughly 18.5% lower than the `6D`
ceiling. The flash to `6D` used `nvflash` and the `92.00.6D.00.0A` ROM file. The ROM
file matches the cards' native subsystem ID (`10DE:1585`) and device ID (`0x20C2`).
This match makes the operation a same-model flash, not a cross-model flash.

## Flashing procedure

Follow these steps in order.

1. Stop every container and process that uses the GPUs. Confirm this with
   `nvidia-smi --query-compute-apps`. The output must be empty.
2. Unload the nvidia driver modules: `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset`, and
   `nvidia`. This step lets `nvflash` call `mmap()` on the card's BAR directly. Without
   this step, `nvflash` fails with an `mmap(): /dev/mem` error.
3. Run `nvflash -i<N> --save backup.rom` for each target card. Do this before you
   change anything else.
4. Run `nvflash -i<N> <target>.rom`. Confirm the version and ID match before you
   accept the prompt.
5. Reboot the server. The new VBIOS has no effect until the reboot.

`nvflash` reads its yes-or-no confirmation from the controlling TTY. It does not read
stdin. If you pipe input to `nvflash` over a non-interactive SSH command, the pipe has
no effect. Use a real pty for this, for example `ssh -tt` or a pty-driving wrapper.
