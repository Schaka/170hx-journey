# 170HX journey

My own record of an 8× NVIDIA CMP 170HX inference box: the hardware, the kernel I
had to build myself, the drivers, and the model-serving stack running on top. See
[AGENTS.md](AGENTS.md) for what this repo is (and isn't) — short version: it's my
notes, not a guide for anyone else's setup.

- [HARDWARE.md](HARDWARE.md) — cards, PLX switches, risers, cabling
- [KERNEL.md](KERNEL.md) — the custom kernel and driver stack, and why a stock kernel
  can't do this
- [BIOS.md](BIOS.md) — VBIOS state across all 8 cards
- [SERVING.md](SERVING.md) — the model-serving stack (DeepSeek-V4-Flash, Qwen3.8) and
  agent-client config
- [scripts/](scripts/) — the actual launcher scripts running on the workstation

## Thanks

None of the P2P/BAR1 work here would exist without
[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) — see
[KERNEL.md](KERNEL.md) for exactly what it made possible and why it's required, not
optional, on this hardware. Also referencing
[cachenetics/170tune](https://github.com/cachenetics/170tune) and
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) throughout.
