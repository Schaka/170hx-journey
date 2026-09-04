# 170HX journey

This is my own record of an inference server with eight NVIDIA CMP 170HX cards. It
covers the hardware, the kernel I built myself, the drivers, and the model-serving
stack that runs on top. See [AGENTS.md](AGENTS.md) for what this repository is and is
not. In short: these are my notes, not a guide for your setup.

- [HARDWARE.md](HARDWARE.md): the cards, the PLX switches, the risers, and the cables.
- [KERNEL.md](KERNEL.md): the custom kernel and driver stack, and why a stock kernel
  cannot do this job.
- [BIOS.md](BIOS.md): the VBIOS state on all eight cards.
- [SERVING.md](SERVING.md): the model-serving stack (DeepSeek-V4-Flash, Qwen3.8) and
  the agent-client configuration.
- [compose/](compose/): the compose file that starts each model server.
- [scripts/](scripts/): the launcher scripts that run on the workstation.

## Thanks

The P2P and BAR1 work here depends on a project:
[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker).
See [KERNEL.md](KERNEL.md) for what that project made possible.
I need it on this hardware. I do not just want it.

This repository also refers to two more projects:
[cachenetics/170tune](https://github.com/cachenetics/170tune) and
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport).
