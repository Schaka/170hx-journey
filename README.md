# 170HX journey

This is my own record of an inference server with eight NVIDIA CMP 170HX cards. It
covers the hardware, the kernel I built myself, the drivers, and the model-serving
stack that runs on top. These are my notes for my own use, not a guide for your setup,
and not a support project. Nothing here assumes it works on hardware other than mine.

Every file describes the system as it stands today, not how it got that way. When
something on the box changes, I edit the file to match. I do not add a note about the
change. `git log` and `git blame` already keep that history.

- [HARDWARE.md](HARDWARE.md): the cards, the PLX switches, the risers, and the cables.
- [KERNEL.md](KERNEL.md): the custom kernel and driver stack, and why a stock kernel
  cannot do this job.
- [BIOS.md](BIOS.md): the VBIOS state on all eight cards.
- [SERVING.md](SERVING.md): the model-serving stack (DeepSeek-V4-Flash, Qwen3.8,
  and GLM-5.3, across several backends and quantizations) and the agent-client
  configuration.
- [compose/](compose/): the compose file that starts each model server.
- [scripts/](scripts/): the launcher scripts that run on the workstation.

## Thanks

- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) (v0.3):
  the base unlock tool for the CMP 170HX. It restores full SM compute throughput and
  unlocked HBM2e memory geometry. It is also the driver source lineage this box runs
  today, up to `610.57.04`.
- [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker): the P2P patch set on
  top of that base. See [KERNEL.md](KERNEL.md) for exactly what it made possible on
  this hardware.
- [cachenetics/170tune](https://github.com/cachenetics/170tune): general 170HX
  tuning reference.
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport): the Qwen3.8,
  GLM-5.3, and alternative DeepSeek-V4 serving stack in [SERVING.md](SERVING.md).
