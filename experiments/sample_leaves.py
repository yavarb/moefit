"""Exclusive (self) time per leaf frame for one thread of a macOS `sample` report.

usage: python3 sample_leaves.py sample.txt THREAD_LINE_START THREAD_LINE_END
Prints leaf symbol -> samples, plus coarse buckets for oMLX decode analysis.
"""
import re, sys
from collections import Counter

path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(path).read().splitlines()[a - 1:b]
pat = re.compile(r"^([ +!:|]*)(\d+) (.*)$")
rows = []
for ln in lines:
    m = pat.match(ln)
    if m:
        rows.append((len(m.group(1)), int(m.group(2)), m.group(3)))
leaf = Counter()
# a node's self time = count - sum(children counts)
for i, (d, c, s) in enumerate(rows):
    kids = 0
    j = i + 1
    while j < len(rows) and rows[j][0] > d:
        if rows[j][0] == min(r[0] for r in rows[i + 1:j + 1] if r[0] > d):
            pass
        j += 1
    # direct children: depth == first child's depth
    if i + 1 < len(rows) and rows[i + 1][0] > d:
        cd = rows[i + 1][0]
        kids = sum(r[1] for r in rows[i + 1:j] if r[0] == cd)
    self_c = c - kids
    if self_c > 0:
        name = re.sub(r"\s+\(in .*$", "", s).split("  ")[0][:90]
        leaf[name] += self_c
tot = sum(leaf.values())
print("total", tot)
for k, v in leaf.most_common(25):
    print(f"{v:6d} {100*v/tot:5.1f}%  {k}")
