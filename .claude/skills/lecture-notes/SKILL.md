---
name: lecture-notes
description: Write a CS231n lecture chapter for this Quarto book from the Stanford video and slide deck — fetch and clean the transcript, map it against the deck, draft flowing prose with maths and cited papers, and render. Use when the user asks to write, draft, or rewrite notes for a numbered lecture.
---

# Writing a CS231n lecture chapter

The product is a chapter of a Quarto book that a student reads instead of re-watching
the lecture. It is prose, not an outline of prose. Someone should be able to read it
end to end and come away with the argument, not just the vocabulary.

Argue for the ideas in the order that makes them make sense, which is not always the
order Fei-Fei used on the day. The lecture is the source; the chapter is the artefact.

## The constraint that shapes everything: context

A deck is 50-300 pages and a transcript is an hour of speech. Reading either one end
to end, once, costs more than the entire rest of the job — and having read it, you
then drag it through every subsequent turn. So: **never read a transcript or a deck
in full.** Both get reduced to a skimmable map first, and only the slices you have
identified a need for get pulled into context.

Target for a whole lecture, ingestion through render: **60k tokens**. If you are past
that at the halfway point, you are reading too widely — go back to the maps.

## Phase 1 — Ingest (costs nothing; it is all shell)

```sh
tools/fetch_lecture.sh <N>                       # transcript + minute index
python3 tools/slides_text.py slides/2025/lecture_<N>.pdf   # deck text + contents map
```

`fetch_lecture.sh` resolves the lecture number against `transcripts/playlist.tsv`, so
no URL is needed; if the user supplies one anyway, check it matches. Lecture 1 has two
decks (`lecture_1_part_1.pdf`, `lecture_1_part_2.pdf`) — run the extractor on each.

YouTube blocks this network unauthenticated, so yt-dlp uses `--cookies-from-browser
chrome` and needs `deno` on PATH. If a fetch fails on a bot check, that is what broke;
say so rather than retrying variations.

This produces, per lecture:

| File | What it is | Read it? |
| --- | --- | --- |
| `transcripts/lecture-NN.txt` | one sentence per line, `[HH:MM:SS]` stamped | **in slices only** |
| `transcripts/lecture-NN.index.txt` | one line per minute | yes, whole — it is ~1k tokens |
| `slides/2025/text/lecture_N.map` | one line per slide | yes, whole — it is ~1k tokens |
| `slides/2025/text/lecture_N.txt` | every slide's text, page-marked | **in slices only** |

All four are gitignored. They are Stanford's material: they stay on disk, they never
get committed, and their wording never lands in the chapter (see *Register*, below).

## Phase 2 — Map, then plan (~4k tokens)

Read the minute index and the deck map — only those two. Together they tell you what
the lecture covers and roughly where, which is enough to plan the chapter.

Write `outlines/lecture-NN.md` (gitignored, a working file) holding one row per
intended section: the section's title, the transcript range that carries it, the slide
pages that carry it, and a sentence on the argument it has to make. This table is the
thing you consult while drafting; it is why you do not need the full transcript.

Decide what to cut here, not later. Course logistics, grading, office hours, guest
speaker introductions, the assignment schedule, jokes, live demos that do not survive
as text, and anything that reduces to "we will cover this in week 9" are out. A
student reading these notes two years from now needs the ideas. Roughly a third of a
lecture is usually admin and repetition; cutting it is the job, not a shortcut.

Show the user the section plan before drafting. It is cheap to redirect at this stage
and expensive afterwards.

## Phase 3 — Draft, one section at a time (~3k tokens per section)

For each section in the outline, and only then, pull its slices:

```sh
awk '/^\[00:14:/,/^\[00:22:/' transcripts/lecture-01.txt      # the transcript range
python3 tools/slides_pages.py slides/2025/text/lecture_1_part_1.txt 19-24
```

Render a slide to an image only when the contents map says the page has no text, or
when a diagram's layout is itself the point:

```sh
python3 tools/slides_pages.py slides/2025/lecture_1_part_1.pdf 22 --png --outdir <scratchpad>
```

Write the section, then move to the next. Do not hold six sections' worth of source in
context at once.

### Register

Formal, continuous prose. The reader is a capable student, not a skimmer.

Bullet points are the default failure mode and they are what makes a page read as
machine-written. A list is correct for genuinely enumerable things — a set of
hyperparameters, the steps of an algorithm, the layers of a named architecture. It is
wrong for ideas that stand in relation to one another, and ideas in relation are most
of what a lecture contains. When you catch yourself writing three parallel bullets,
you have three sentences that belong in a paragraph, and the paragraph will be better
because it has to state how they connect.

Paragraphs carry the argument forward. Each one should follow from the last by
something stronger than adjacency — a consequence, a tension, an objection, a
narrowing. The seams are where the reader either follows you or stops.

Things to avoid, all of which read as generated: opening a section by restating its
own title; "In this section, we will explore"; "It is important to note that";
"Let's dive in"; closing summaries that repeat what was just said; and the
three-adjective rhythm ("powerful, flexible, and efficient"). Prefer the concrete
claim to the hedged one. Write "AlexNet halved the error rate", not "AlexNet
demonstrated significant improvements in performance".

Never paraphrase the transcript line by line. Understand the point, then make it in
your own words, in the order your chapter needs. The transcript is evidence of what
was taught; it is not a draft.

### Maths, and the things maths needs

Put the mathematics in. A chapter on linear classifiers that never writes
$f(x, W) = Wx + b$ has not done its job. Display equations for anything load-bearing,
inline for anything referred to in passing, and define every symbol on first use — the
sentence after an equation should say what the reader is looking at.

```markdown
$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$ {#eq-svm-loss}
```

KaTeX is pinned to 0.18.1 and renders on the client; `\begin{aligned}`, `\mathbb`,
`\operatorname` are all available. Reference equations as `@eq-svm-loss`.

Architectures, loss surfaces, and the shape of a tensor as it moves through a network
are worth a diagram far more often than they get one. When the diagram is structural
rather than photographic, draw it as an SVG in `figures/<chapter>/diagram_NN.svg`
rather than asking for an image — a hand-drawn diagram matches the book's palette and
carries no provenance problem. The palette is viridis; see the theme files.

### Citations

When the notes reach a result that has a canonical paper behind it, link the paper the
first time it matters, on the phrase a reader would search for:

> the architecture that won ImageNet in 2012 ([AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html))

Link the official venue — arXiv abstract page, the proceedings entry, the journal DOI.
Never a blog post, a Medium summary, or a PDF mirror on someone's course page.

Be sparing. A paragraph with four links is a paragraph the reader stops reading. Aim
for the handful of papers a student would actually go and read; a lecture that names
twenty papers usually deserves five links and a References section carrying the rest.

When you need a fact the lecture does not establish — a date, a parameter count, an
error rate — check it against the paper or the official documentation before writing
it down. Do not assert numbers from memory, and do not use a secondary source where
the primary one exists.

### Figures you cannot draw

Photographs, dataset samples, real model outputs and historical images have to come
from somewhere, and this repository does not take them from the slides — see
`figures/SOURCES.md` for why, and for the provenance table every figure lands in.

When a section needs one, stop and ask the user, giving them something they can paste
straight into an image search:

> **Нужна картинка:** рецептивные поля из эксперимента Хьюбела и Визеля.
> **Запрос:** `Hubel Wiesel 1959 cat visual cortex simple complex cells receptive field diagram`
> **Что должно быть на картинке:** ориентированная полоса света, реакция нейрона, подпись про простые и сложные клетки.
> **Куда:** `figures/01-history/hubel_wiesel_1959.avif`

Batch these — collect the requests for a whole draft and ask once, rather than
interrupting per section. Keep writing around the gap; leave the figure reference in
place so the chapter is complete the moment the file arrives.

**Settle figure provenance during Phase 2, not here.** Some figures already in the
repository cannot be used: `figures/SOURCES.md` records that the AlexNet architecture
figure is reproduced in the CS231n decks under a permission granted to that course
and not to this repository, and the same reasoning covers other publisher figures.
Others are marked `unidentified`, meaning nobody has traced where the bitmap came
from. Neither kind belongs in a new chapter. Read the relevant `SOURCES.md` rows while
planning, because discovering at assembly time that a section's illustration is
unusable means rewriting around a hole you have already written into.

## Phase 4 — Assemble and verify

The chapter is `lectures/NN-slug.qmd`, with front matter matching the existing
chapters, and a line registering it under the right part in `_quarto.yml`. Sections get
`{#sec-slug}` anchors so they can be cross-referenced.

**A figure must stand alone in its own block.** If the prose that follows an
`![...](...){#fig-x}` line ends up on the same line, pandoc renders it as an inline
image rather than a figure, and every `@fig-x` reference silently fails to resolve.
The render only warns; it does not fail. Leave a blank line on both sides of every
figure, without exception.

Check the links before rendering — it is faster than finding them afterwards:

```sh
python3 tools/check_links.py lectures/NN-slug.qmd
```

Do not hand-roll this with `curl`. Academic publishers answer scripts with 403,
repositories answer with 405 and GitHub rate-limits to 429, none of which means a
link is broken; the script knows the difference and a shell loop does not.

Then render and look at it:

```sh
export PATH="$HOME/.local/quarto/bin:$PATH" && quarto render
```

Confirm the maths rendered rather than surviving as raw `$…$`, that every `@fig-` and
`@eq-` resolves, and that no figure reference points at a file that is not there.

When checking images in the browser, **scroll the page first**. Every image except the
first carries `loading="lazy"` courtesy of `filters/lazy-images.lua`, so a check for
`naturalWidth === 0` reports every below-the-fold figure as broken on a fresh load.
Scroll to the bottom, wait, and then count.

Do not commit `_site/`, `transcripts/`, `slides/`, or `outlines/`.

## Phase 5 — Close out

Report what was written, what was deliberately cut from the lecture and why, which
figures are still outstanding, and roughly what the whole thing cost in tokens.

Then, and this is required every run: **propose improvements to this skill.** Keep
them structural rather than incidental — a phase that consistently costs more than it
should, a style rule that keeps needing to be restated because it is not written down
sharply enough, a step that could move from the model to a script, a check that would
have caught something late. One or two real observations beat a list of small ones.
The point is that the skill gets better every lecture, so say what you actually
noticed, including when the honest answer is that a rule here did not survive contact
with the material.
