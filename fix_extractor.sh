#!/bin/bash
# fix_extractor.sh - fixes the run-3b failure (every evolved file had no "class Model").
# Cause: the LLM's reply began with a placeholder code block ("# Explanation goes here"), and the repo's
# clean_code_from_llm() keeps only the FIRST fenced block. Two fixes:
#   1. src/llm_utils.py: keep the fenced block that defines "class Model" (last one if several);
#      if no block does, behave exactly as before.
#   2. templates/TitanicRules.txt: ask for ONE code block and no explanation text.
# Run from anywhere inside the repo checkout:  bash fix_extractor.sh   (writes files only; submits nothing)
set -euo pipefail
cd "$(dirname "$0")"
[ -f src/llm_utils.py ] && [ -f templates/TitanicRules.txt ] || { echo "ERROR: run install_run3.sh first"; exit 1; }

python3 - <<'PY'
p = "src/llm_utils.py"
s = open(p).read()
if "with_model" not in s:
    old = '''    try:
        # Extract and clean code assuming it is enclosed in triple backticks
        return '\\n'.join(code_from_llm.strip().split("```")[1].split('\\n')[1:]).strip()
    except (IndexError, AttributeError) as e:'''
    assert s.count(old) == 1, "clean_code_from_llm not in the expected form"
    new = '''    try:
        text = code_from_llm.strip()
        # Prefer the closed fenced block that defines the evolved class (the last one if several);
        # replies often start with a placeholder or explanation block.
        blocks = re.findall(r"```[^\\n`]*\\n(.*?)```", text, flags=re.S)
        with_model = [b for b in blocks if re.search(r"^\\s*class\\s+Model\\b", b, flags=re.M)]
        if with_model:
            return with_model[-1].strip()
        # Otherwise: original behaviour (first fenced block, minus its first line)
        return '\\n'.join(text.split("```")[1].split('\\n')[1:]).strip()
    except (IndexError, AttributeError) as e:'''
    open(p, "w").write(s.replace(old, new))

open("templates/TitanicRules.txt", "w").write("""Rules:
1. Keep the class name Model and its fit(X, y) and predict(X) methods. Model() is created with no arguments, so every setting needs a default inside the class.
2. X and y are pandas objects. X has 7 numeric columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), already standardized. Use only scikit-learn, numpy and pandas.
3. Reply with exactly one Markdown python code block that holds the complete modified code. Put no text before or after the code block, and do not add example usage, an if __name__ == "__main__" block, or tests.
4. Keep any hyperparameter search small (at most 20 candidates and 3 folds) so training finishes within a minute.
""")
print("patched src/llm_utils.py and templates/TitanicRules.txt")
PY
echo; sed -n '/^def clean_code_from_llm/,/return "ERROR"/p' src/llm_utils.py; echo; cat templates/TitanicRules.txt
