#!/usr/bin/env python3
"""
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ----------------------------- configuration ----------------------------- #
DEFAULT_ROOT = Path(os.environ.get("CHAOS_ROOT", Path.home() / "ChaosSail"))

# TODO: should not be hard-coded
CATEGORIES = {
    1: "development - software",
    2: "routines",
    3: "learning - software",
    4: "learning - hardware",
    5: "research - ",
    6: "wildcard",
}

# ----------------------------- helpers ---------------------------------- #

def iso_today() -> str:
    return dt.date.today().isoformat()


def ensure_base(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "journal.md"
    if not journal.exists():
        journal.write_text("# Chaos Journal\n", encoding="utf-8")


def today_dir(root: Path) -> Path:
    today = iso_today()
    y = today[:4]
    m = today[5:7]
    ddir = root / y / m / today
    for sub in ("spark", "flow", "trace"):
        (ddir / sub).mkdir(parents=True, exist_ok=True)
    return ddir


def append_journal(root: Path, line: str) -> None:
    ensure_base(root)
    with (root / "journal.md").open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def pick_editor() -> List[str]:
    """Return the user's preferred editor as an argv list.

    `EDITOR`/`VISUAL` may contain additional arguments (e.g. ``code -w``),
    so we need to split the string similarly to how a shell would.  The
    previous implementation returned the entire string as a single element,
    causing lookups to fail when an editor command included spaces.
    """
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        return shlex.split(editor)
    if os.name == "nt":
        return ["notepad"]
    return ["vim"]


# ----------------------------- journal parsing -------------------------- #
@dataclass
class JEntry:
    date: dt.date
    kind: str  # spark|flow|trace|other
    text: str
    tags: Tuple[str, ...]

TAG_RE = re.compile(r"#[A-Za-z0-9_]+")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def parse_journal(root: Path) -> List[JEntry]:
    journal = root / "journal.md"
    if not journal.exists():
        return []
    entries: List[JEntry] = []
    for line in journal.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = DATE_RE.match(line)
        if not m:
            continue
        d = dt.date.fromisoformat(m.group(1))
        kind = "other"
        if " spark:" in line:
            kind = "spark"
        elif " flow:" in line:
            kind = "flow"
        elif " trace:" in line:
            kind = "trace"
        tags = tuple(t.lower().lstrip("#") for t in TAG_RE.findall(line))
        entries.append(JEntry(d, kind, line, tags))
    return entries


def known_tags(entries: Sequence[JEntry]) -> Counter:
    c: Counter = Counter()
    for e in entries:
        c.update(e.tags)
    return c


# ----------------------------- commands --------------------------------- #

def cmd_init(args: argparse.Namespace) -> int:
    root: Path = args.root
    ensure_base(root)
    print(f"Initialized Chaos Sail at: {root}")
    return 0


def cmd_spark(args: argparse.Namespace) -> int:
    root: Path = args.root
    ddir = today_dir(root)
    ts = dt.datetime.now().strftime("%H%M%S")
    fpath = ddir / "spark" / f"{ts}.txt"

    text = " ".join(args.text).strip() if args.text else ""
    if not text:
        subprocess.run(pick_editor() + [str(fpath)], check=False)
    else:
        fpath.write_text(text + "\n", encoding="utf-8")

    print(f"[spark] {fpath}")
    append_journal(root, f"{iso_today()} | spark: {fpath.name}")
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    root: Path = args.root
    ddir = today_dir(root)
    ts = dt.datetime.now().strftime("%H%M%S")
    name = (args.name or "chunk").strip()
    safe = "-".join(name.split()) if name else "chunk"
    fpath = ddir / "flow" / f"{ts}-{safe}.md"

    header = f"# flow: {name}\n\n"
    fpath.write_text(header, encoding="utf-8")
    subprocess.run(pick_editor() + [str(fpath)], check=False)

    print(f"[flow] {fpath}")
    append_journal(root, f"{iso_today()} | flow: {fpath.name}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    root: Path = args.root
    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 2
    ddir = today_dir(root)
    ts = dt.datetime.now().strftime("%H%M%S")
    dest = ddir / "trace" / f"{ts}-{src.name}"
    shutil.copy2(src, dest)

    msg = (args.message or "").strip()
    print(f"[trace] {dest}")
    jline = f"{iso_today()} | trace: {dest.name}"
    if msg:
        jline += f" | {msg}"
    append_journal(root, jline)
    return 0


def cmd_roll(args: argparse.Namespace) -> int:
    n = random.randint(1, 6)
    print(f"Dice: {n} ({CATEGORIES[n]})")
    return 0


# ----------------------------- stats ------------------------------------ #

def within_period(d: dt.date, since: Optional[int], start: Optional[dt.date], end: Optional[dt.date]) -> bool:
    if since is not None:
        cutoff = dt.date.today() - dt.timedelta(days=since)
        return d >= cutoff
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def current_streak(active_days: Sequence[dt.date]) -> int:
    s = set(active_days)
    streak = 0
    day = dt.date.today()
    while day in s:
        streak += 1
        day -= dt.timedelta(days=1)
    return streak


def longest_streak(active_days: Sequence[dt.date]) -> int:
    s = set(active_days)
    best = 0
    for d in sorted(s):
        if d - dt.timedelta(days=1) not in s:
            k = 0
            cur = d
            while cur in s:
                k += 1
                cur += dt.timedelta(days=1)
            best = max(best, k)
    return best


def cmd_stats_all(args: argparse.Namespace) -> int:
    root: Path = args.root
    entries = parse_journal(root)
    if not entries:
        print("No journal entries yet.")
        return 0

    # Filter by time window
    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    filt = [e for e in entries if within_period(e.date, args.since, start, end)]
    if not filt:
        print("No entries in the selected period.")
        return 0

    kinds = Counter(e.kind for e in filt)
    days = sorted({e.date for e in filt})
    active = len(days)
    span_days = (max(days) - min(days)).days + 1 if days else 0
    ratio = (kinds.get("trace", 0) / kinds.get("flow", 1)) if kinds.get("flow", 0) else float('inf')

    tags = known_tags(filt)
    top_tags = ", ".join(f"#{t}({c})" for t, c in tags.most_common(10)) or "—"

    print("=== Chaos Sail: stats-all ===")
    if args.since is not None:
        print(f"Period: last {args.since} days")
    else:
        print(f"Period: {start or '-'} .. {end or '-'}")
    print(f"Active days: {active} / span {span_days} days (unique days with any entry)")
    print(f"Counts: spark={kinds.get('spark',0)} flow={kinds.get('flow',0)} trace={kinds.get('trace',0)}")
    if kinds.get("flow", 0):
        print(f"trace/flow ratio: {ratio:.2f}")
    else:
        print("trace/flow ratio: n/a (no flow)")
    print(f"Current streak: {current_streak(days)} days | Longest streak: {longest_streak(days)} days")
    print(f"Top tags: {top_tags}")

    return 0


def match_tags(existing: Counter, raw_tags: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (matched, unknown) where matched are normalized existing tags.
    Accepts prefixes; resolves unambiguous prefixes.
    """
    norm = {t.lower(): c for t, c in existing.items()}
    matched: List[str] = []
    unknown: List[str] = []
    for arg in raw_tags:
        q = arg.lower().lstrip('#')
        if q in norm:
            matched.append(q)
            continue
        # prefix match
        cands = [t for t in norm.keys() if t.startswith(q)]
        if len(cands) == 1:
            matched.append(cands[0])
        elif len(cands) == 0:
            unknown.append(arg)
        else:
            # ambiguous; print suggestions and treat as unknown for now
            print(f"ambiguous tag '{arg}': {', '.join('#'+t for t in sorted(cands)[:10])} ...", file=sys.stderr)
            unknown.append(arg)
    return matched, unknown


def cmd_stats_tags(args: argparse.Namespace) -> int:
    root: Path = args.root
    entries = parse_journal(root)
    if not entries:
        print("No journal entries yet.")
        return 0
    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None

    tags_counter = known_tags(entries)
    if not args.tags:
        # list top tags
        print("Known tags (by frequency):")
        for t, c in tags_counter.most_common(50):
            print(f"#{t}\t{c}")
        return 0

    wanted, unknown = match_tags(tags_counter, args.tags)
    if unknown:
        print("Unknown/ambiguous: " + ", ".join(unknown), file=sys.stderr)
        print("Use `chaos_sail.py tags` to see available tags.")
        return 2

    # Filter entries by tags
    def has_tags(e: JEntry) -> bool:
        s = set(e.tags)
        if args.logic == "and":
            return all(t in s for t in wanted)
        return any(t in s for t in wanted)

    filt = [e for e in entries if within_period(e.date, args.since, start, end) and has_tags(e)]
    if not filt:
        print("No entries match the criteria.")
        return 0

    kinds = Counter(e.kind for e in filt)
    days = sorted({e.date for e in filt})
    tags = known_tags(filt)

    print("=== Chaos Sail: stats-tags ===")
    print("Tags:", ", ".join(f"#{t}" for t in wanted))
    if args.since is not None:
        print(f"Period: last {args.since} days")
    else:
        print(f"Period: {start or '-'} .. {end or '-'}")
    print(f"Entries: {len(filt)} (spark={kinds.get('spark',0)} flow={kinds.get('flow',0)} trace={kinds.get('trace',0)})")
    print(f"Active days: {len(days)}")
    print("Top co-tags:", ", ".join(f"#{t}({c})" for t, c in tags.most_common(10)))

    if args.show:
        print("--- last entries ---")
        # Show last N lines with date and a trimmed body
        for e in list(filt)[-args.show:]:
            print(e.text)
    return 0


def cmd_tags(args: argparse.Namespace) -> int:
    root: Path = args.root
    entries = parse_journal(root)
    tags_counter = known_tags(entries)
    for t, c in tags_counter.most_common():
        print(f"#{t}\t{c}")
    return 0


# ----------------------------- completion ------------------------------- #

def complete_from_env(args: argparse.Namespace) -> int:
    """Very small bash completion handler for `stats-tags`.
    Usage (installed by `install-completion`):
      complete -o default -C "/path/to/chaos_sail.py __complete" chaos_sail.py
    Bash will call this command with COMP_LINE/COMP_POINT in env.
    We return newline-separated candidates.
    """
    line = os.environ.get("COMP_LINE", "")
    point_str = os.environ.get("COMP_POINT")
    point = int(point_str) if point_str and point_str.isdigit() else len(line)
    line = line[:point]
    parts = line.split()
    if not parts or "stats-tags" not in parts:
        return 0

    cur = ""
    if not line.endswith(" "):
        cur = parts[-1]
    # Collect known tags
    tags = [t for t, _ in known_tags(parse_journal(args.root)).most_common()]
    # We complete bare names (no '#')
    cur_norm = cur.lstrip('#').lower()
    cands = [t for t in tags if t.startswith(cur_norm)]
    sys.stdout.write("\n".join(cands))
    return 0


def cmd_install_completion(args: argparse.Namespace) -> int:
    script_path = Path(sys.argv[0]).expanduser().resolve()
    line = f'complete -o default -C "{script_path} __complete" chaos_sail.py\n'
    rc = Path.home() / ".bashrc"
    if args.print:
        print(line, end="")
        return 0
    with rc.open("a", encoding="utf-8") as f:
        f.write("\n# Chaos Sail completion\n")
        f.write(line)
    print(f"Bash completion installed into {rc}. Restart your shell or `source {rc}`.")
    return 0


# ----------------------------- argparse --------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chaos_sail.py",
        description="Chaos Sail — spark/flow/trace cadence without bureaucracy",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory (default: {DEFAULT_ROOT})",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init", help="initialize the file structure")
    sp_init.set_defaults(func=cmd_init)

    sp_spark = sub.add_parser("spark", help="drop a tiny note into today's spark/")
    sp_spark.add_argument("text", nargs=argparse.REMAINDER, help="note text (optional)")
    sp_spark.set_defaults(func=cmd_spark)

    sp_flow = sub.add_parser("flow", help="open editor for a chunk in today's flow/")
    sp_flow.add_argument("name", nargs="?", help="optional name for the chunk")
    sp_flow.set_defaults(func=cmd_flow)

    sp_trace = sub.add_parser("trace", help="copy an artifact into today's trace/")
    sp_trace.add_argument("file", help="path to the file to copy")
    sp_trace.add_argument("message", nargs="?", help="optional short note (you can include #tags)")
    sp_trace.set_defaults(func=cmd_trace)

    sp_roll = sub.add_parser("roll", help="roll a d6 to pick a category")
    sp_roll.set_defaults(func=cmd_roll)

    sp_stats_all = sub.add_parser("stats-all", help="overall statistics")
    sp_stats_all.add_argument("--since", type=int, help="look back N days (overrides --start/--end)")
    sp_stats_all.add_argument("--start", help="start date YYYY-MM-DD")
    sp_stats_all.add_argument("--end", help="end date YYYY-MM-DD")
    sp_stats_all.set_defaults(func=cmd_stats_all)

    sp_stats_tags = sub.add_parser("stats-tags", help="statistics filtered by tags (prefix OK)")
    sp_stats_tags.add_argument("tags", nargs="*", help="tags or prefixes (without #)")
    sp_stats_tags.add_argument("--since", type=int, help="look back N days")
    sp_stats_tags.add_argument("--start", help="start date YYYY-MM-DD")
    sp_stats_tags.add_argument("--end", help="end date YYYY-MM-DD")
    sp_stats_tags.add_argument("--logic", choices=["or", "and"], default="or", help="combine multiple tags")
    sp_stats_tags.add_argument("--show", type=int, default=0, help="print last N matching lines")
    sp_stats_tags.set_defaults(func=cmd_stats_tags)

    sp_tags = sub.add_parser("tags", help="list known tags by frequency")
    sp_tags.set_defaults(func=cmd_tags)

    sp_install = sub.add_parser("install-completion", help="append simple bash completion to ~/.bashrc")
    sp_install.add_argument("--print", action="store_true", help="print the completion line instead of writing")
    sp_install.set_defaults(func=cmd_install_completion)

    sp_complete = sub.add_parser("__complete", help=argparse.SUPPRESS)
    sp_complete.set_defaults(func=complete_from_env)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
