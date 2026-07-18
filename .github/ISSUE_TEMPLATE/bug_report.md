---
name: Bug report
about: Something AC computed, emitted, or installed wrong
title: "[bug]: "
labels: bug
---

**AC version**
Output of `pip show archc | head -2` (or the git commit if running from source):

**Command you ran**
The full command line, e.g.
`ac-compile --hardware h100 --params 7 --tokens 2 ...`

**Hardware target(s)**
e.g. h100 / b200 / tpu_v5p (the AC `--hardware` flag, not your local machine)

**What happened**
Paste the complete error output or the wrong result (full traceback, not a screenshot).

**What you expected**

**Baseline config (if used)**
Attach or paste the `--baseline-config` JSON, or name the reference config
(e.g. `configs/mistral_7b.json`).

**Reproducibility**
- [ ] Fails every run
- [ ] Fails intermittently (please say how often)

**Environment**
OS, Python version (`python --version`), install method (pip / source).
