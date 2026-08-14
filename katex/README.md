# KaTeX 0.18.1

The maths on this site is typeset during the build, not in the reader's
browser. `tools/prerender_math.ts` runs `katex.min.js` from this directory over
the rendered HTML; what a reader downloads is `katex.min.css` and whichever of
the twenty font files their formulas happen to need.

Nothing here is fetched at read time, and nothing is fetched at build time
either. That is the point: before this, the version was pinned in `_quarto.yml`
so that a KaTeX release could not silently change how a page looked. Committing
the library keeps that guarantee and adds a second one — the build no longer
depends on a CDN being reachable or honest.

| File            | Role                      | Shipped to readers |
| --------------- | ------------------------- | ------------------ |
| `katex.min.js`  | typesets during the build | no                 |
| `katex.min.css` | styles the result         | yes                |
| `fonts/*.woff2` | the twenty KaTeX faces    | on demand          |

`project.resources` in `_quarto.yml` lists the stylesheet and the fonts, and
deliberately does not list the JavaScript, so Quarto never copies it into the
site.

KaTeX is MIT licensed; see [LICENSE](LICENSE).

## Provenance

From the npm tarball rather than the CDN, so the download can be checksummed:

```sh
curl -sSL -o katex.tgz https://registry.npmjs.org/katex/-/katex-0.18.1.tgz
# sha256  7e6100b7fe6439ba91d918d8cb2873171a9fdec979281d508959cf5f7dba1da8
tar xzf katex.tgz
cp package/dist/katex.min.js package/dist/katex.min.css package/LICENSE .
cp package/dist/fonts/*.woff2 fonts/
```

`katex.min.js` and `katex.min.css` in that tarball are byte for byte what
`cdn.jsdelivr.net/npm/katex@0.18.1/dist/` was serving, verified by sha256 at the
time they were committed:

```
68b9115510b8cedb9909a10de7799c94c0707481296f755c0a8888cb8fcde216  katex.min.js
0fb711c9c74cb1718661933948b653fbc09a627da5dde8926b4d10585370993e  katex.min.css
```

So the formulas are set by exactly the code the site was already using.

## The one modification

Upstream's `katex.min.css` offers each face as `woff2`, then `woff`, then
`ttf`. Only the `woff2` files are committed — every browser that can render
this site has supported `woff2` since 2016 — so the two dead alternatives are
stripped from each `src`, which also keeps the stylesheet from naming files
that are not there:

```sh
perl -i -pe 's/,url\(fonts\/[^)]+\.(?:woff|ttf)\) format\("(?:woff|truetype)"\)//g' katex.min.css
```

Re-run that after any update. Nothing else here is edited.
