# Repo intent

This repository documents my journey. I built an inference server with eight NVIDIA
CMP 170HX cards. This repository records the hardware, the kernel, the drivers, and
the model-serving stack on top of that hardware. This repository exists so that I have
one current record of how this machine works and why it works that way. This
repository is not a support project. This repository is not a guide for other people.
This repository does not try to work on other people's hardware. Read this repository
as a personal log that contains working configuration, not as documentation for an
audience.

## Writing rule: status quo only, no history

Every file in this repository describes the current state of the system. Never write
about a past state, a change, or the reason an old approach failed. Use `git log` and
`git blame` for that. A document that says "we used to do X but changed to Y" is wrong
the moment you write it. When something changes, edit the document to describe the new
state. Do not add a note about the change.

## Writing rule: Simple English

Every document in this repository, and every future document, must follow the Simple
English rules from
[AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) in Plain mode. This
plugin is installed for Claude Code as `simple-english@simple-english`.

Before you write or edit any document in this repository, load the `simple-english`
skill and apply Plain mode. The core rules:

- Write short sentences: 20 words for an instruction, 25 words for an explanation.
- Use active voice and simple tenses. Name the actor.
- Use only these modals: can, will, must.
- Use one word for one meaning, through the whole document.
- Define a new technical term the first time you use it, in under ten words.
- Delete words that add no fact. Use the word-swap table in the skill for examples.
- Do not use semicolons or em dashes. Write two sentences instead.
- Never change code, commands, file paths, flags, or quoted error text.

If the `simple-english` skill or plugin is not available in your environment, read
`prompts/system-prompt.md` from the [AminBlg/SimpleEnglish
repository](https://github.com/AminBlg/SimpleEnglish) and follow it instead.

## Scope

This repository will grow as the setup changes: new kernel patches, new serving
configuration, new hardware. All new material belongs here, written as the current
state at the time you write it.

## For coding agents working in this repository

- Do not add historical notes, changelogs, or "this used to be" comments to any
  document. See the history rule above.
- Do not assume anything here works on other hardware. The kernel arguments, PCI bus
  IDs, and card counts belong to this one machine.
- Load the `simple-english` skill before you write or edit any document.
