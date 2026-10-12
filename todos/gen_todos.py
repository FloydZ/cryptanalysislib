#!/usr/bin/env python3
"""Collect TODO/FIXME/XXX/HACK comments from tracked files into todos/*.md."""
import os, re, sys, subprocess, collections, datetime
root = sys.argv[1]
out_dir = os.path.join(root, 'todos')
files = subprocess.run(['git', '-C', root, 'ls-files'], capture_output=True, text=True).stdout.split('\n')
files = [f for f in files if f and not re.match(r'(deps/|build|\.claude/|benchmark/|todos/)', f)]
MARK = re.compile(r'\b(todo|fixme)\b|\b(XXX|HACK)\b', re.I)
COMMENT = re.compile(r'(//|/\*|^\s*\*|#)')
def area(f):
    p = f.split('/')
    if p[0] == 'src': return 'src-' + p[1].replace('.h', '').replace('.cpp', '') if len(p) > 2 else 'src-toplevel'
    if p[0] in ('tests', 'bench'): return p[0]
    if p[0] == 'cryptanalysislib': return 'python'
    return 'other'
entries = collections.defaultdict(list)
total = 0
for f in files:
    path = os.path.join(root, f)
    try:
        lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    except (IsADirectoryError, FileNotFoundError):
        continue
    md = f.endswith(('.md', '.py'))   # docs and python docstrings: any line
    for i, line in enumerate(lines):
        m = MARK.search(line)
        if not m: continue
        if m.group(2) and m.group(2) != m.group(2).upper(): continue   # xxx/hack only uppercase
        c = COMMENT.search(line)
        if not md and (not c or c.start() > m.start()): continue        # marker must be inside a comment
        text = line[m.end():].strip(' :-*/\t')
        if len(text) < 12:   # bare marker: add the next line as context
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
            ctx = nxt.lstrip('/*# ').strip()
            if ctx: text = (text + ' — ' if text else '') + ('`' + ctx[:100] + '`' if not COMMENT.match(nxt) else ctx[:140])
        text = text.replace('|', '\\|')
        entries[area(f)].append((f, i + 1, m.group(0), text or '(no description)'))
        total += 1
os.makedirs(out_dir, exist_ok=True)
for a, es in entries.items():
    with open(os.path.join(out_dir, a + '.md'), 'w') as o:
        o.write(f'# TODOs: {a}\n\n{len(es)} entries. Format: `- [ ] file:line [MARKER] text`.\n')
        cur = None
        for f, l, mk, t in es:
            if f != cur:
                o.write(f'\n## {f}\n\n'); cur = f
            o.write(f'- [ ] `{f}:{l}` **{mk.upper()}** {t}\n')
with open(os.path.join(out_dir, 'README.md'), 'w') as o:
    o.write('# TODO index\n\n')
    o.write(f'Generated {datetime.date.today()} from `git ls-files` (excluding `deps/`, `build*`, `benchmark/`, `.claude/`).\n')
    o.write('Collected markers: TODO, FIXME (any case), XXX, HACK, only inside comments (any line in .md and .py files).\n')
    o.write(f'Regenerate: `python3 todos/gen_todos.py .`\n\n**{total} entries** in {sum(len({e[0] for e in v}) for v in entries.values())} files.\n\n')
    if os.path.exists(os.path.join(out_dir, 'known-issues.md')):
        o.write('Open issues that are not TODO comments: [known-issues.md](known-issues.md) (hand-maintained).\n\n')
    o.write('| area | entries | files |\n|---|---:|---:|\n')
    for a, es in sorted(entries.items(), key=lambda kv: -len(kv[1])):
        o.write(f'| [{a}]({a}.md) | {len(es)} | {len({e[0] for e in es})} |\n')
print(total)
