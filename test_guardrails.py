"""Negative test: deliberately corrupt main.tex in memory and confirm each
guardrail in verify_paper.py actually fires. A check that never triggers is
worthless."""
import io, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "overleaf", "main.tex")
orig = io.open(SRC, encoding="utf-8").read()

CASES = [
    ("banned phrase",
     lambda s: s.replace("Both learned models beat the resource baseline",
                         "Our proposed model outperforms every baseline and beat the resource baseline"),
     "banned"),
    ("wrong number",
     lambda s: s.replace("286{,}621 Indian Premier League", "999{,}999 Indian Premier League"),
     "number"),
    ("table row drift",
     lambda s: s.replace("MI & 41 & 48.8 [34, 63]", "MI & 41 & 71.2 [34, 63]"),
     "table"),
    ("spine reorder",
     lambda s: s.replace("First, \\emph{model selection}", "First, \\emph{rating stability was measured and model selection}"),
     "spine"),
    # must strip the WHOLE sentence: leaving any interval attached to the same
    # sentence legitimately satisfies the guardrail
    ("unhedged small-n claim",
     lambda s: re.sub(r"CSK's overall win rate is.*?does not\.",
                      "CSK win 56.0\\\\,\\\\% of their matches and their bowling "
                      "contributes more than their batting.", s, flags=re.S),
     "hedge"),
]

failures = []
for name, mutate, expect in CASES:
    mutated = mutate(orig)
    if mutated == orig:
        failures.append(f"{name}: mutation did not apply (anchor text changed?)")
        continue
    io.open(SRC, "w", encoding="utf-8").write(mutated)
    try:
        r = subprocess.run([sys.executable, "-X", "utf8",
                            os.path.join(HERE, "verify_paper.py")],
                           capture_output=True, text=True, cwd=HERE)
        out = r.stdout + r.stderr
        fired = r.returncode != 0 and f"[{expect}]" in out
        print(f"{'PASS' if fired else 'FAIL'}  {name:26} -> expected [{expect}]")
        if not fired:
            failures.append(f"{name}: guardrail [{expect}] did not fire\n{out[:400]}")
    finally:
        io.open(SRC, "w", encoding="utf-8").write(orig)

# confirm the file was restored byte-for-byte
assert io.open(SRC, encoding="utf-8").read() == orig, "main.tex NOT restored!"
print("\nmain.tex restored intact")
print("RESULT:", "all guardrails fire" if not failures else "PROBLEMS:\n  " + "\n  ".join(failures))
sys.exit(1 if failures else 0)
