# Fonts

The site self-hosts its type. Nothing here is fetched from a third party at
runtime, so reading the notes sends no request to Google.

| Family                                                           | Role                                 | Licence                          |
| ---------------------------------------------------------------- | ------------------------------------ | -------------------------------- |
| [Space Grotesk](https://github.com/floriankarsten/space-grotesk) | headings, the hero, UI chrome        | [OFL 1.1](OFL-SpaceGrotesk.txt)  |
| [Source Serif 4](https://github.com/adobe-fonts/source-serif)    | long-form prose                      | [OFL 1.1](OFL-SourceSerif4.txt)  |
| [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono)     | section numbers, figure labels, code | [OFL 1.1](OFL-JetBrainsMono.txt) |

All three are unmodified upstream builds, redistributed under the SIL Open Font
License 1.1 with the copyright notices intact. The licence permits bundling them
with any software as long as they are not sold on their own.

## Regenerating

The `.woff2` files are the subsetted, variable builds the Google Fonts css2 API
serves to a current Chrome. To refresh them, request the CSS with a modern
browser user agent and download what it points at:

```sh
curl -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
  "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700\
&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600\
&family=JetBrains+Mono:wght@400;500&display=swap"
```

Without that user agent the API falls back to `.ttf`, which is roughly four
times the size.

Only the `latin`, `latin-ext` and `greek` subsets are kept — `latin-ext` for the
accented author names in citations, `greek` for the odd symbol in running prose
(real maths is set by KaTeX, which ships its own fonts). Cyrillic and Vietnamese
were dropped: no page contains either script.

Each family is a variable font, so one file per subset covers every weight the
design uses. `fonts.css` declares weight _ranges_ rather than a face per weight;
Source Serif 4 also carries an optical-size axis, which browsers apply through
`font-optical-sizing: auto`.

`fonts.css` is wired up in `_quarto.yml` under `format.html.css`, and the
directory is listed under `project.resources` so Quarto copies the `.woff2`
files into `_site/`.
