#!/usr/bin/env python3
"""Turn a YouTube VTT caption track into a line-per-sentence transcript with timecodes.

The caption tracks Stanford ships are cue-wrapped at arbitrary points mid-sentence,
which makes them useless for reading and for quoting a time range. This rejoins the
cues into sentences, stamps each one with the time its first word was spoken, and
writes a file that greps and slices cleanly.

    python3 tools/transcript_clean.py transcripts/raw/lecture-01.en-US.vtt \
        --out transcripts/lecture-01.txt \
        --title "Lecture 1: Introduction" --url https://youtu.be/2fq9wYslV0A

Add --index to also write a one-line-per-minute skim map next to the transcript,
which is what you read first to decide which slices you actually need.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

CUE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
TAG_RE = re.compile(r"<[^>]+>")
SENT_END_RE = re.compile(r"[.!?][\"')\]]*$")


def parse_vtt(text: str) -> list[tuple[float, str]]:
    """Return [(start_seconds, cue_text)], deduplicated and stripped of markup."""
    cues: list[tuple[float, str]] = []
    start: float | None = None
    buf: list[str] = []

    def flush() -> None:
        if start is None:
            return
        line = " ".join(buf).strip()
        line = TAG_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and (not cues or cues[-1][1] != line):
            cues.append((start, line))

    for raw in text.splitlines():
        line = raw.strip()
        m = CUE_RE.match(line)
        if m:
            flush()
            h, mi, s, ms = (int(x) for x in m.groups()[:4])
            start, buf = h * 3600 + mi * 60 + s + ms / 1000, []
        elif line and start is not None and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            buf.append(line)
    flush()
    return cues


def to_sentences(cues: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Rejoin cues into sentences, keeping the timestamp of each sentence's first cue."""
    out: list[tuple[float, str]] = []
    start: float | None = None
    parts: list[str] = []
    for t, text in cues:
        if start is None:
            start = t
        parts.append(text)
        if SENT_END_RE.search(text):
            out.append((start, " ".join(parts)))
            start, parts = None, []
    if parts and start is not None:
        out.append((start, " ".join(parts)))
    return out


def hms(seconds: float) -> str:
    """Format a second offset as HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def build_index(sentences: list[tuple[float, str]], words: int = 14) -> list[str]:
    """One line per minute of lecture: the timecode and the first few words spoken in it."""
    rows, seen = [], set()
    for t, text in sentences:
        minute = int(t) // 60
        if minute in seen:
            continue
        seen.add(minute)
        rows.append(f"[{hms(minute * 60)}] {' '.join(text.split()[:words])}")
    return rows


def main() -> int:
    """Parse arguments, convert the caption track, and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtt", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--index", action="store_true", help="also write <out stem>.index.txt")
    args = ap.parse_args()

    if not args.vtt.is_file():
        print(f"no such caption file: {args.vtt}", file=sys.stderr)
        return 1

    sentences = to_sentences(parse_vtt(args.vtt.read_text(encoding="utf-8", errors="replace")))
    if not sentences:
        print("parsed zero sentences — is this really a VTT track?", file=sys.stderr)
        return 1

    header = [
        f"# {args.title or args.vtt.stem}",
        f"# source: {args.url}" if args.url else "",
        f"# {len(sentences)} sentences, runtime {hms(sentences[-1][0])}",
        "# Stanford course material — local working copy, never commit or redistribute.",
        "",
    ]
    body = [f"[{hms(t)}] {text}" for t, text in sentences]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join([h for h in header if h != ""] + [""] + body) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(sentences)} sentences, runtime {hms(sentences[-1][0])}")

    if args.index:
        idx = args.out.with_suffix(".index.txt")
        idx.write_text("\n".join(build_index(sentences)) + "\n", encoding="utf-8")
        print(f"{idx}: {len(build_index(sentences))} minute markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
