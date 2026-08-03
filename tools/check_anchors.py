#!/usr/bin/env python3
"""Check the structural invariants Quarto only warns about.

    tools/check_anchors.py                     # every chapter plus index.qmd
    tools/check_anchors.py lectures/06-x.qmd   # one file, still resolving
                                               # references against the whole book

`quarto render` reports all four of these failures as WARNING and then produces
a page anyway, so a broken chapter looks like a successful build. Worse, a
single duplicate identifier poisons Quarto's crossref index for the whole book:
one collision silently turns a dozen unrelated `@sec-`/`@eq-` references into
unresolved text. The checks are:

1. Every `$$ … $$ {#eq-…}` block is contiguous. A blank line inside display
   math splits it into two blocks and the label attaches to neither, so the
   equation loses its number and every `@eq-` pointing at it dangles.
2. A blank line precedes every heading. Without one, pandoc folds the heading
   into the preceding paragraph and its `{#sec-…}` anchor never exists.
3. Every `@sec-`/`@eq-`/`@fig-`/`@tbl-` reference resolves to a definition
   somewhere in the book.
4. No identifier is defined twice across the book.

Exit status is 1 if anything failed, so this is safe to run before a commit.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFINE = re.compile(r"\{#((?:sec|eq|fig|tbl)-[a-z0-9][a-z0-9-]*)[^}]*\}")
REFER = re.compile(r"(?<![\w`])@((?:sec|eq|fig|tbl)-[a-z0-9][a-z0-9-]*[a-z0-9])")
HEADING = re.compile(r"^#{1,6} \S")
FENCE = re.compile(r"^\s*(```|:::)")


def book_files():
    """Every source page, in book order as far as it matters here."""
    files = sorted((ROOT / "lectures").glob("*.qmd"))
    index = ROOT / "index.qmd"
    if index.exists():
        files.insert(0, index)
    return files


def strip_code(lines):
    """Blank out fenced code blocks so their contents are never parsed."""
    out, in_fence = [], False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def check_math(path, lines, fail):
    """Display-math blocks must be contiguous and must close."""
    open_at = None
    for n, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if stripped == "$$":
            if open_at is None:
                open_at = n
            else:
                open_at = None  # an unlabelled block, closed
        elif stripped.startswith("$$ {#") and stripped.endswith("}"):
            if open_at is None:
                fail(f"{path}:{n}: display math closes without an opening $$")
            else:
                body = lines[open_at : n - 1]
                if any(not ln.strip() for ln in body):
                    label = DEFINE.search(stripped)
                    name = label.group(1) if label else "?"
                    fail(
                        f"{path}:{open_at}-{n}: blank line inside the $$ block "
                        f"labelled {name}; the label will not attach"
                    )
                open_at = None
    if open_at is not None:
        fail(f"{path}:{open_at}: display math opens and never closes")


def check_headings(path, lines, fail):
    """A heading folded into the paragraph above it loses its anchor."""
    for n, line in enumerate(lines, 1):
        if not HEADING.match(line) or n == 1:
            continue
        previous = lines[n - 2]
        if previous.strip() and not FENCE.match(previous):
            fail(
                f"{path}:{n}: no blank line before heading {line.strip()!r}; "
                "pandoc will fold it into the paragraph above"
            )


def main(argv):
    """Run every check over the named files, or over the whole book."""
    files = book_files()
    targets = [pathlib.Path(a).resolve() for a in argv[1:]] or files

    failures = []

    def fail(message):
        failures.append(message)

    # Definitions are collected book-wide even when one file is being checked,
    # because both the duplicate check and the reference check are global.
    defined = {}
    for path in files:
        for n, line in enumerate(strip_code(path.read_text().splitlines()), 1):
            for name in DEFINE.findall(line):
                where = f"{path.relative_to(ROOT)}:{n}"
                if name in defined:
                    fail(
                        f"{where}: identifier {name!r} is already defined at "
                        f"{defined[name]}; a duplicate breaks crossref resolution "
                        "for the whole book, not just this reference"
                    )
                else:
                    defined[name] = where

    for path in targets:
        rel = path.relative_to(ROOT)
        raw = path.read_text().splitlines()
        lines = strip_code(raw)
        check_math(rel, raw, fail)
        check_headings(rel, lines, fail)
        for n, line in enumerate(lines, 1):
            for name in REFER.findall(line):
                if name not in defined:
                    fail(f"{rel}:{n}: reference @{name} has no definition")

    checked = ", ".join(str(p.relative_to(ROOT)) for p in targets)
    if failures:
        for message in failures:
            print(message)
        print(f"\n{len(failures)} problem(s) in {checked}")
        return 1
    print(f"{len(defined)} identifiers, no problems in {checked}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
