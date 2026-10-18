"""Static sanity checks on main.tex (no LaTeX installation required)."""
import re, io, os, collections, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overleaf"))
s = io.open("main.tex", encoding="utf-8").read()
fail = []

def strip_comments(text):
    """Drop LaTeX comment text so it cannot skew brace/environment counts."""
    out = []
    for line in text.split("\n"):
        j, cut = 0, None
        while j < len(line):
            if line[j] == "\\":
                j += 2; continue
            if line[j] == "%":
                cut = j; break
            j += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)

body = strip_comments(s)

bad = [ch for ch in s if ord(ch) > 127]
print("non-ASCII chars:", len(bad), bad[:10])
if bad: fail.append("non-ascii")

beg = collections.Counter(re.findall(r"\\begin\{([A-Za-z]+\*?)\}", s))
end = collections.Counter(re.findall(r"\\end\{([A-Za-z]+\*?)\}", s))
mism = {k: (beg[k], end[k]) for k in set(beg) | set(end) if beg[k] != end[k]}
print("environment mismatch:", mism or "none")
if mism: fail.append("envs")

depth, i, bad_brace = 0, 0, False
while i < len(body):
    if body[i] == "\\":
        i += 2; continue
    if body[i] == "{":
        depth += 1
    elif body[i] == "}":
        depth -= 1
        if depth < 0:
            bad_brace = True; break
    i += 1
print("brace balance:", "OK" if depth == 0 and not bad_brace else f"BAD (net {depth})")
if depth or bad_brace: fail.append("braces")

susp = []
for n, L in enumerate(s.split("\n"), 1):
    j = 0
    while j < len(L):
        if L[j] == "\\":
            j += 2; continue
        if L[j] == "%":
            if L[:j].strip():
                susp.append((n, L.strip()[:70]))
            break
        j += 1
print("possible unescaped %%:", susp or "none")
if susp: fail.append("percent")

labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print("dangling ref:", (refs - labels) or "none")
if refs - labels: fail.append("refs")

bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
cites = {c.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", s) for c in grp.split(",")}
print("dangling cite:", (cites - bibs) or "none")
print("uncited bibitems:", (bibs - cites) or "none")
if cites - bibs: fail.append("cites")

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)
missing = [f for f in figs if not os.path.exists(os.path.join("figures", f))]
print(f"figures referenced: {len(figs)}, missing: {missing or 'none'}")
if missing: fail.append("figures")

print("remaining placeholders:", re.findall(r"\\ph\{([^}]*)\}", s))
print("\nRESULT:", "FAIL -> " + ", ".join(fail) if fail else "all static checks passed")
sys.exit(1 if fail else 0)
