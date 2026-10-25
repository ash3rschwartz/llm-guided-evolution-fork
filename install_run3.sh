#!/bin/bash
# install_run3.sh - installs the improved prompts, the cross-validated eval.py and the held-out scoring script.
# Run from anywhere inside the repo checkout:  bash install_run3.sh
# It only writes files and edits two lines of src/cfg/constants_titanic.py. It submits nothing.
set -euo pipefail
cd "$(dirname "$0")"

[ -f sota/Titanic/eval.py ] || { echo "ERROR: run this from the llm-guided-evolution-fork checkout"; exit 1; }
[ -f src/cfg/constants_titanic.py ] || { echo "ERROR: src/cfg/constants_titanic.py not found"; exit 1; }

mkdir -p templates/titanic/improved

# ---- 1. rules appended to every prompt (replaces the contradictory ConstantRules.txt for this run)
cat > templates/TitanicRules.txt <<'__END_RULES__'
Rules:
1. Keep the class name Model and its fit(X, y) and predict(X) methods. Model() is created with no arguments, so every setting needs a default inside the class.
2. X and y are pandas objects. X has 7 numeric columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), already standardized. Use only scikit-learn, numpy and pandas.
3. Return the full modified code in one Markdown python code block. Do not add example usage, an if __name__ == "__main__" block, tests, or anything after the code block.
4. Keep any hyperparameter search small (at most 20 candidates and 3 folds) so training finishes within a minute.
5. Write at most two sentences of explanation before the code block.
__END_RULES__

# ---- 2. the six improved prompts
cat > templates/titanic/improved/1.txt <<'__END_P1__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Modify the model to reduce the false positive rate below 0.06. It is acceptable for the false negative rate to rise to about 0.40.

The current code block:
```python
{} 
```
__END_P1__
cat > templates/titanic/improved/2.txt <<'__END_P2__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Modify the model to reduce the false negative rate below 0.20. It is acceptable for the false positive rate to rise to about 0.26.

The current code block:
```python
{} 
```
__END_P2__
cat > templates/titanic/improved/3.txt <<'__END_P3__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Replace or extend the classifier with a different scikit-learn model family (for example GradientBoostingClassifier, ExtraTreesClassifier, LogisticRegression, KNeighborsClassifier or SVC) that may trade the two error rates differently. Keep fit(X, y) and predict(X).

The current code block:
```python
{} 
```
__END_P3__
cat > templates/titanic/improved/4.txt <<'__END_P4__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Modify the model to find a better balance, lowering both error rates compared with the seed.

The current code block:
```python
{} 
```
__END_P4__
cat > templates/titanic/improved/5.txt <<'__END_P5__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Make predict use predict_proba (or decision_function) with a decision threshold chosen inside fit, for example by cross-validation on the training data, so that the false positive rate is lower. Do not pass the threshold as a constructor argument.

The current code block:
```python
{} 
```
__END_P5__
cat > templates/titanic/improved/6.txt <<'__END_P6__'
This code is a binary classifier for Titanic survival. It is scored on two objectives, both minimized: the false positive rate (the share of passengers who died that it predicts survived) and the false negative rate (the share of survivors that it predicts died). The seed model has a false positive rate of about 0.13 and a false negative rate of about 0.30.

Use class weights and/or a lower decision threshold, set inside the class, so that the false negative rate is lower while the false positive rate stays below about 0.26.

The current code block:
```python
{} 
```
__END_P6__

# ---- 3. eval.py: cross-validated fitness on the training rows (--mode test = final held-out check)
cat > sota/Titanic/eval.py <<'__END_EVAL__'
import argparse
import importlib
import os

from pathlib import Path as p
from os.path import join as pj
import sys


import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

# Same partition as our earlier ML / MOGP assignments: 33% held out, seed 100.
SPLIT_SEED = 100
TEST_SIZE = 0.33
# Evolution scores every individual by stratified K-fold cross-validation on the TRAINING rows only.
# The held-out rows are never touched in --mode cv, so they stay an unbiased final test.
N_FOLDS = 5


def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)    

    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    
    return save_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path where the trained model will be saved")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed to use for deterministic evaluation")
    parser.add_argument('--variant_dir', type=str, default='models', help="directory where models will be written by LLM-GE")
    parser.add_argument('--mode', type=str, default='cv', choices=['cv', 'test'],
                        help="cv: score by cross-validation on the training rows (used during evolution). "
                             "test: train on all training rows and score on the held-out rows (final check only).")
    return parser.parse_args()


def scale(X_fit, *others):
    """StandardScaler fit on X_fit only; every frame comes back as a DataFrame with its columns/index."""
    scaler = StandardScaler()
    out = [pd.DataFrame(scaler.fit_transform(X_fit), columns=X_fit.columns, index=X_fit.index)]
    for X in others:
        out.append(pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index))
    return out


def fp_fn(y_true, y_pred):
    matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(matrix[0, 1]), int(matrix[1, 0])


if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)

    args = get_args()

    # Set the random seeds for deterministic results
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # This is LLM Guided Code
    # Import the module dynamically
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)



    # this will get the gene id value 
    try:
        gene_id = args.model.split('model_')[1] # Causing an error
    except:
        gene_id = 'seed'
    save_dir = f'{args.save_dir}/{gene_id}' 
    create_save_dir(save_dir)


    train_df = pd.read_csv('data/processed_train.csv')
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)

    if args.mode == 'test':
        # Final check: train on all training rows, score on the held-out rows.
        X_fit, X_eval = scale(X_train, X_test)
        model = model_module.Model()
        model.fit(X_fit, y_train)
        fp_count, fn_count = fp_fn(y_test, model.predict(X_eval))
    else:
        # Evolution fitness: out-of-fold FP and FN counts over the training rows.
        # A fresh Model() and a scaler fit on the fold's training part are used for every fold.
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
        fp_count, fn_count = 0, 0
        for fit_idx, val_idx in skf.split(X_train, y_train):
            X_fit, X_val = scale(X_train.iloc[fit_idx], X_train.iloc[val_idx])
            model = model_module.Model()
            model.fit(X_fit, y_train.iloc[fit_idx])
            fp, fn = fp_fn(y_train.iloc[val_idx], model.predict(X_val))
            fp_count += fp
            fn_count += fn

    # Print out FP, FN (first line of output; scripts parse this)
    print(fp_count, fn_count)
    """
    This line formats the results into comma-separated string
    fp_count: Number of false positives.
    fn_count: Number of false negatives.
    """
    results_text = f"fp_count,fn_count\n{fp_count},{fn_count}"

    # Define Output Filename
    """
    Defines an Output Filename
    gene_id is a unique identifier for the experiment or model.
    Evolution (cv) results go to results/ where LLM-GE reads them; final held-out (test)
    scores go to test_results/ so they can never be mistaken for fitness.
    """
    if args.mode == 'test':
        filename = os.path.abspath(f'test_results/{gene_id}_test.csv')
    else:
        filename = os.path.abspath(f'results/{gene_id}_results.csv')
    
    dir_path = os.path.dirname(filename)

    # Create the directory, ignore error if it already exists
    os.makedirs(dir_path, exist_ok=True)

    # Open the file in write mode and write the text
    with open(filename, 'w') as file:
        file.write(results_text)

    print(f"results have been written to {filename}")

    print('='*120);print('job done');print('='*120)
__END_EVAL__

# ---- 4. script that scores the final front on the held-out rows
cat > score_front_on_test.py <<'__END_SCORE__'
"""Re-score the final LLM-GE Pareto front on the held-out test rows.

During evolution every individual is scored by cross-validation on the training rows only
(sota/Titanic/eval.py --mode cv), so the held-out rows are untouched. This script is the
single, final look at them: for each individual on the front it trains on all training rows
and counts false positives / false negatives on the held-out rows (eval.py --mode test).

Run from the repo root, on a compute node, after analyze_llmge.py:
    uv run python score_front_on_test.py
Output: analysis_out/llmge_front_test.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

N_NEG_TEST, N_POS_TEST = 172, 123   # held-out split: 172 non-survivors, 123 survivors


def run_eval(eval_py, variant_dir, model_name, save_dir):
    cmd = [sys.executable, eval_py, "--mode", "test", "--model", model_name,
           "--variant_dir", variant_dir, "--save_dir", save_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40 * 60)
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = first.split()
    if proc.returncode == 0 and len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1]), ""
        except ValueError:
            pass
    tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return None, None, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default="analysis_out/llmge_front.csv")
    ap.add_argument("--out", default="analysis_out/llmge_front_test.csv")
    ap.add_argument("--eval", default="sota/Titanic/eval.py")
    ap.add_argument("--variant-dir", default="sota/Titanic/models/llmge_models")
    args = ap.parse_args()

    eval_py = os.path.abspath(args.eval)
    variant_dir = os.path.abspath(args.variant_dir)
    with open(args.front) as fh:
        front = list(csv.DictReader(fh))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        # the seed model first, for reference ("model" is the seed file in sota/Titanic)
        fp, fn, err = run_eval(eval_py, variant_dir, "model", tmp)
        rows.append({"gene_id": "seed", "gen": "", "cv_fp": "", "cv_fn": "", "test_fp": fp, "test_fn": fn, "error": err})
        for r in front:
            fp, fn, err = run_eval(eval_py, variant_dir, f"model_{r['gene_id']}", tmp)
            rows.append({"gene_id": r["gene_id"], "gen": r["gen"], "cv_fp": r["fp"], "cv_fn": r["fn"],
                         "test_fp": fp, "test_fn": fn, "error": err})
            print(f"{r['gene_id']}: cv ({r['fp']}, {r['fn']}) -> test ({fp}, {fn}) {err}", flush=True)

    header = ["gene_id", "gen", "cv_fp", "cv_fn", "test_fp", "test_fn", "test_fpr", "test_fnr", "error"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            ok = r["test_fp"] is not None
            r["test_fpr"] = r["test_fp"] / N_NEG_TEST if ok else ""
            r["test_fnr"] = r["test_fn"] / N_POS_TEST if ok else ""
            w.writerow(r)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
__END_SCORE__

# ---- 5. point the run at the new prompt group and rules file
CFG=src/cfg/constants_titanic.py
sed -i 's#^ISLAND_CONTROLLER_PROMPT_GROUPS = .*#ISLAND_CONTROLLER_PROMPT_GROUPS = "titanic/improved"#' "$CFG"
if ! grep -q '^CONSTANT_RULES_PATH' "$CFG"; then
  sed -i '/^ISLAND_CONTROLLER_PROMPT_GROUPS = /a CONSTANT_RULES_PATH = "templates/TitanicRules.txt"' "$CFG"
fi

echo
echo "Installed. Changed lines in $CFG:"
grep -n '^ISLAND_CONTROLLER_PROMPT_GROUPS\|^CONSTANT_RULES_PATH' "$CFG"
echo
echo "New prompt files:"
ls templates/titanic/improved
