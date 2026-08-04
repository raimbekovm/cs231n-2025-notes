#!/usr/bin/env bash
# Fetch and clean the caption track for one CS231n lecture.
#
#   tools/fetch_lecture.sh 2            # resolve lecture 2 from tools/playlist.tsv
#   tools/fetch_lecture.sh 2 --force    # re-download even if the transcript exists
#
# Writes transcripts/lecture-NN.txt (one sentence per line, timecoded) and
# transcripts/lecture-NN.index.txt (one line per minute). Both are gitignored:
# they are Stanford's material and stay local.
#
# YouTube blocks unauthenticated requests from this network, so yt-dlp is run with
# --cookies-from-browser chrome. macOS may raise a Keychain prompt the first time.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

N="${1:?usage: fetch_lecture.sh <lecture-number> [--force]}"
FORCE="${2:-}"
NN="$(printf '%02d' "$N")"
PLAYLIST="$REPO/tools/playlist.tsv"
OUT="$REPO/transcripts/lecture-$NN.txt"

for tool in yt-dlp deno; do
  command -v "$tool" >/dev/null || { echo "missing $tool — see the skill's setup notes" >&2; exit 1; }
done
[ -f "$PLAYLIST" ] || { echo "no $PLAYLIST — refresh it with the playlist command in SKILL.md" >&2; exit 1; }

if [ -f "$OUT" ] && [ "$FORCE" != "--force" ]; then
  echo "$OUT already exists — pass --force to re-download"
  exit 0
fi

row="$(awk -F'\t' -v n="$NN" '$1 == n {print; exit}' "$PLAYLIST")"
[ -n "$row" ] || { echo "lecture $NN is not in $PLAYLIST" >&2; exit 1; }
VID="$(printf '%s' "$row" | cut -f2)"
TITLE="$(printf '%s' "$row" | cut -f4)"
URL="https://youtu.be/$VID"
echo "==> $TITLE  ($URL)"

mkdir -p "$REPO/transcripts/raw"
# --ignore-no-formats-error: yt-dlp still runs format selection under --skip-download,
# so when YouTube's n-challenge solver fails it finds no video streams and aborts before
# writing the subtitles it already located. The captions are unaffected; only the abort is.
yt-dlp --cookies-from-browser chrome --skip-download --ignore-no-formats-error \
       --write-subs --write-auto-subs --sub-langs "en.*" --sub-format "vtt/best" \
       -o "$REPO/transcripts/raw/lecture-$NN.%(ext)s" "$URL" 2>&1 | grep -vE '^(WARNING|\[download\])' || true

# Prefer a human-authored track: it is punctuated and roughly a fifth the size of the
# auto-generated one, which repeats every line as its word timings roll in.
VTT=""
for cand in "en-US" "en-GB" "en-en-US" "en-orig" "en"; do
  f="$REPO/transcripts/raw/lecture-$NN.$cand.vtt"
  [ -f "$f" ] || continue
  if [ -z "$VTT" ] || [ "$(wc -c <"$f")" -lt "$(wc -c <"$VTT")" ]; then VTT="$f"; fi
done
[ -n "$VTT" ] || { echo "no caption track was downloaded for lecture $NN" >&2; exit 1; }
echo "==> using $(basename "$VTT")"

python3 "$REPO/tools/transcript_clean.py" "$VTT" --out "$OUT" --title "CS231n 2025 — $TITLE" --url "$URL" --index
