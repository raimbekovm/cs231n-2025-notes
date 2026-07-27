#!/usr/bin/env python3
"""Check that every external link in a chapter resolves, without the false alarms.

A naive curl loop is useless here. Academic publishers return 403 to anything that
does not look like a browser, so Wiley, OUP and science.org all report "broken" for
links that are perfectly fine. DOIs are the case that matters, and a DOI is healthy
when doi.org redirects it to a publisher — whatever the publisher then says to a
script is not evidence about the link.

    python3 tools/check_links.py lectures/01-introduction.qmd

Exit status is non-zero if anything is genuinely broken, so it can gate a render.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import urllib.error
import urllib.request

LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)|<(https?://[^>\s]+)>")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
# A 403 from these means "no scripts please", not "no such page".
BOT_WALLED = ("doi.org", "wiley.com", "academic.oup.com", "science.org", "springer.com", "nature.com")


def probe(url: str, timeout: int = 25) -> tuple[str, str]:
    """Return (verdict, detail) for one URL. Verdict is ok, redirect, or BROKEN."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return "ok", f"{resp.status} {resp.url[:70]}"
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            # Rate limiting says something about how often we asked, not about the link.
            return "skip", "429 (rate-limited; not checked)"
        if exc.code == 405:
            # Method Not Allowed is the server objecting to how it was asked, not
            # reporting a missing page. Repositories like DSpace do this routinely.
            return "ok", "405 (server refused the method; the page is there)"
        if exc.code in (401, 403) and any(h in url or h in (exc.url or "") for h in BOT_WALLED):
            return "ok", f"{exc.code} (publisher blocks scripts; link itself resolves)"
        return "BROKEN", f"HTTP {exc.code}"
    except Exception as exc:
        return "BROKEN", type(exc).__name__


def main() -> int:
    """Extract links from the given files, probe each once, and report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    urls: set[str] = set()
    for f in args.files:
        for m in LINK_RE.finditer(f.read_text(encoding="utf-8", errors="replace")):
            urls.add(m.group(1) or m.group(2))

    broken = 0
    for url in sorted(urls):
        verdict, detail = probe(url)
        if verdict == "BROKEN":
            broken += 1
        print(f"{verdict:7s} {url}\n        {detail}")

    print(f"\n{len(urls)} links, {broken} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
