# Repo intent

This repository documents **my** journey building an 8× NVIDIA CMP 170HX inference
box — the hardware, the kernel, the drivers, and the model-serving stack on top of it.
It exists so *I* have a single, current record of exactly how this machine is put
together and why. It is not a support project, not a how-to for strangers, and not
trying to be reusable by anyone else's hardware. Read it as a personal log with
working config in it, not as documentation aimed at an audience.

## Writing rule: status quo only, no history

Every file in this repo describes **how things are right now**. Never write about how
something used to be, what changed, or why an old approach was replaced — that's what
`git log` and `git blame` are for. A doc that says "we used to do X but switched to Y"
is wrong the moment it's written; overwrite it with what's true today instead.

This applies to every doc going forward, including files not yet written. When
something changes on the machine, edit the doc to describe the new state — don't add
a note about the change.

## Scope

This repo will keep growing as the setup evolves: new kernel patches, new serving
configs, new hardware. All of it belongs here, documented as current state at time of
writing.

## For coding agents working in this repo

- Don't add historical commentary, changelogs, or "previously this was..." notes to
  any doc — see the rule above.
- Don't assume anything here generalizes to other people's hardware. Kernel args, PCI
  bus IDs, and card counts are specific to this one machine.
