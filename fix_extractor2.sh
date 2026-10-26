#!/bin/bash
# fix_extractor2.sh - second fix for the run-3c failure (38 of 89 individuals saved the PROMPT as their code).
# Cause: the LLM sometimes writes malformed fences, e.g. "```python" immediately followed by "```from sklearn...",
# and ends with a stray "```python". No valid fenced block exists, so clean_code_from_llm() returned "" and
# llm_mutation.augment_network() then wrote the whole prompt text into the model file (SyntaxError).
# Two changes (run fix_extractor.sh first):
#   1. src/llm_utils.py: if no clean fenced block defines "class Model", strip all fence markers, take everything
#      from the first top-level import/from/class line, and use it only if it contains "class Model" and parses.
#   2. src/llm_mutation.py: if nothing usable comes back (empty or "ERROR"), keep the PARENT chunk instead of
#      writing the prompt text (or the word ERROR) into the file.
# Run from anywhere inside the repo checkout:  bash fix_extractor2.sh   (writes files only; submits nothing)
set -euo pipefail
cd "$(dirname "$0")"
[ -f src/llm_utils.py ] && [ -f src/llm_mutation.py ] || { echo "ERROR: run from the repo root"; exit 1; }

python3 - <<'PY'
p = "src/llm_utils.py"
s = open(p).read()
if "Malformed fences" not in s:
    old = '''        # Otherwise: original behaviour (first fenced block, minus its first line)
        return '\\n'.join(text.split("```")[1].split('\\n')[1:]).strip()'''
    assert s.count(old) == 1, "run fix_extractor.sh first (clean_code_from_llm not in the expected form)"
    new = '''        # Malformed fences (e.g. "```python" followed directly by "```from ...", or a stray fence glued to the
        # last line): drop every fence marker and take everything from the first top-level import/from/class line.
        import ast as _ast
        flat = re.sub(r"```", "\\n", re.sub(r"```[A-Za-z]*[ \\t]*\\n", "\\n", text))
        lines = flat.split("\\n")
        start = next((i for i, l in enumerate(lines) if re.match(r"(from|import|class)\\s", l)), None)
        if start is not None:
            candidate = "\\n".join(lines[start:]).strip()
            if re.search(r"^\\s*class\\s+Model\\b", candidate, flags=re.M):
                try:
                    _ast.parse(candidate)
                    return candidate
                except SyntaxError:
                    pass
        # Otherwise: original behaviour (first fenced block, minus its first line)
        return '\\n'.join(text.split("```")[1].split('\\n')[1:]).strip()'''
    open(p, "w").write(s.replace(old, new))

p = "src/llm_mutation.py"
s = open(p).read()
if "No usable code from the LLM" not in s:
    old = '''    if not code_from_llm:
        code_from_llm = txt2llm
    else:'''
    assert s.count(old) == 1, "augment_network not in the expected form"
    new = '''    if not code_from_llm or code_from_llm.strip() == "ERROR":
        # the original repo wrote the whole prompt text (txt2llm) here, which can never run
        print("No usable code from the LLM. Falling back to parent chunk.", flush=True)
        code_from_llm = code2llm.strip()
    else:'''
    open(p, "w").write(s.replace(old, new))
print("patched src/llm_utils.py and src/llm_mutation.py")
PY
python3 -m py_compile src/llm_utils.py src/llm_mutation.py && echo "both files compile"
