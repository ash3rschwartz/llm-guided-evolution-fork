#!/bin/bash
# switch_to_holdout.sh - run 3b: new prompts + rules, but fitness scored the same way as run 2
# (train on the 596 training rows, count FP/FN on the 295 held-out rows).
# Run from anywhere inside the repo checkout:  bash switch_to_holdout.sh   (writes files only; submits nothing)
set -euo pipefail
cd "$(dirname "$0")"
[ -f sota/Titanic/eval.py ] && [ -f templates/titanic/improved/1.txt ] || { echo "ERROR: run install_run3.sh first"; exit 1; }

python3 - <<'PY'
import glob

# ---- eval.py: add a 'holdout' mode (the run-2 scoring) and make it the default
p = "sota/Titanic/eval.py"
s = open(p).read()
if "'holdout'" not in s:
    old_args = "default='cv', choices=['cv', 'test'],"
    assert s.count(old_args) == 1
    s = s.replace(old_args, "default='holdout', choices=['cv', 'holdout', 'test'],")
    old_help = '''help="cv: score by cross-validation on the training rows (used during evolution). "'''
    assert s.count(old_help) == 1
    s = s.replace(old_help, '''help="holdout (default): train on the training rows, score on the held-out rows, as in run 2. "
                             "cv: cross-validation on the training rows only. "''')
    old_branch = "    if args.mode == 'test':\n        # Final check"
    assert s.count(old_branch) == 1
    s = s.replace(old_branch, "    if args.mode in ('test', 'holdout'):\n        # holdout/test")
    open(p, "w").write(s)

# ---- prompts: seed rates as scored on the held-out rows, plus label/anti-collapse guard
for f in sorted(glob.glob("templates/titanic/improved/*.txt")):
    t = open(f).read()
    if "Label 1 means survived" in t:
        continue
    old = "The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30."
    assert t.count(old) == 1, f
    new = ("Label 1 means survived and label 0 means did not survive; do not flip the labels, and do not lower one error "
           "rate by predicting nearly everyone as a single class. "
           "The seed model has a false positive rate of about 0.12 and a false negative rate of about 0.27.")
    open(f, "w").write(t.replace(old, new))
print("patched eval.py and prompts")
PY
echo; grep -n "mode" sota/Titanic/eval.py | head -12; echo; cat templates/titanic/improved/2.txt
