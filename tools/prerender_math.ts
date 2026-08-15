// Typeset every formula once, here, instead of in each reader's browser.
//
// Quarto's arrangement is to ship KaTeX to the browser and let it walk the page
// on DOMContentLoaded. That is 74 KB over the wire, a request to a third-party
// CDN, and — on a lecture carrying 242 formulas — a long block of main-thread
// work that only starts after the reader can already see the text, so the page
// reflows under them.
//
// None of it is necessary for a static site: the formulas are the same on every
// visit. This runs the same KaTeX, at the same pinned 0.18.1, over the rendered
// HTML and writes the result into the page. Then it takes the two CDN tags out
// of the head and points the stylesheet at the copy under katex/. What a reader
// downloads is that stylesheet and the handful of font files their formulas
// actually use.
//
//     quarto run tools/prerender_math.ts
//
// It runs on the JavaScript engine Quarto already carries, so the build needs
// nothing installed that rendering the site did not already need.
//
// If this step is skipped the page is exactly what Quarto produced and still
// renders its maths from the CDN, just the slow way. That is the whole reason
// `html-math-method` in `_quarto.yml` still names katex and still pins a URL.

const REPO = new URL("..", import.meta.url).pathname;
const KATEX = `${REPO}katex/katex.min.js`;

// `--site` only exists so the fallback paths can be exercised against a throwaway
// copy; a render always uses the default.
const siteFlag = Deno.args.indexOf("--site");
const SITE = (siteFlag === -1 ? `${REPO}_site` : Deno.args[siteFlag + 1]).replace(/\/$/, "");

// Pandoc puts the bare TeX in as the span's only child, with no `\(`..`\)`
// delimiters — which is why Quarto's script could hand `firstChild.data`
// straight to KaTeX. `[^<]*` both matches that and makes the pass idempotent:
// once a formula holds KaTeX's markup it can no longer match.
const MATH = /<span class="math (inline|display)">([^<]*)<\/span>/g;

// A `.math` span whose content does not start with a tag is TeX nobody has
// typeset. After the pass above there should be none — but the matcher is
// deliberately narrow, so a shape it does not recognise would leave one behind,
// and a page that is half typeset is worse than one that is not typeset at all:
// the script Quarto injects reads `firstChild.data` with no guard, so the first
// already-rendered formula throws and every formula after it stays raw. When
// this finds anything the page is left exactly as Quarto made it.
const UNRENDERED =
  /<span\b(?:[^>"']|"[^"]*"|'[^']*')*\bclass="(?:[^"]*\s)?math(?:\s[^"]*)?"(?:[^>"']|"[^"]*"|'[^']*')*>(?!\s*<)/;

// The three things Quarto emits to typeset in the browser.
const KATEX_SCRIPT = /[ \t]*<script[^>]*\ssrc="[^"]*katex\.min\.js"[^>]*><\/script>\n?/;
const KATEX_BOOTSTRAP =
  /[ \t]*<script>\s*document\.addEventListener\("DOMContentLoaded"[\s\S]*?katex\.render[\s\S]*?<\/script>\n?/;
const KATEX_STYLESHEET = /<link rel="stylesheet" href="[^"]*katex\.min\.css">/;

// The options are copied from the script Quarto was injecting, so what the
// build produces is what the browser was producing.
const OPTIONS = { throwOnError: false, macros: {}, fleqn: false };

const ENTITIES: [string, string][] = [
  ["&lt;", "<"],
  ["&gt;", ">"],
  ["&quot;", '"'],
  ["&#39;", "'"],
  ["&amp;", "&"], // last, so an escaped ampersand is not expanded twice
];

function unescapeHtml(text: string): string {
  return ENTITIES.reduce((out, [entity, char]) => out.replaceAll(entity, char), text);
}

// katex.min.js is a UMD bundle: finding neither CommonJS nor AMD, it assigns
// itself to the global, which is what we want.
async function loadKatex() {
  new Function(await Deno.readTextFile(KATEX)).call(globalThis);
  // deno-lint-ignore no-explicit-any
  return (globalThis as any).katex;
}

async function* pages(dir: string): AsyncGenerator<string> {
  for await (const entry of Deno.readDir(dir)) {
    const path = `${dir}/${entry.name}`;
    if (entry.isDirectory) yield* pages(path);
    else if (entry.name.endsWith(".html")) yield path;
  }
}

try {
  await Deno.stat(SITE);
} catch {
  Deno.exit(0); // nothing rendered, so nothing to typeset
}

const katex = await loadKatex();
let files = 0;
let formulas = 0;
let failures = 0;
let skipped = 0;

for await (const page of pages(SITE)) {
  const html = await Deno.readTextFile(page);
  let typeset = 0;

  let out = html.replace(MATH, (whole, mode: string, tex: string) => {
    try {
      const rendered = katex.renderToString(unescapeHtml(tex), {
        ...OPTIONS,
        displayMode: mode === "display",
      });
      typeset += 1;
      return `<span class="math ${mode}">${rendered}</span>`;
    } catch (error) {
      // `throwOnError: false` already renders bad TeX in red rather than
      // throwing, so getting here means something structural. Keep the formula
      // as it was — the page falls back to rendering it in the browser — and
      // fail the build so nobody finds out from the live site.
      console.error(`prerender_math: ${page}: ${error}`);
      failures += 1;
      return whole;
    }
  });

  if (typeset === 0) continue;

  if (UNRENDERED.test(out)) {
    console.error(
      `prerender_math: ${page} still holds maths this pass does not recognise; ` +
        `leaving the page for the browser to typeset. Widen MATH to cover it.`,
    );
    skipped += 1;
    continue;
  }

  const depth = page.slice(SITE.length + 1).split("/").length - 1;
  const prefix = "../".repeat(depth);
  out = out
    .replace(KATEX_SCRIPT, "")
    .replace(KATEX_BOOTSTRAP, "")
    .replace(KATEX_STYLESHEET, `<link rel="stylesheet" href="${prefix}katex/katex.min.css">`);

  await Deno.writeTextFile(page, out);
  files += 1;
  formulas += typeset;
}

console.log(
  `prerender_math: ${formulas} formulas in ${files} pages` +
    (skipped ? `, ${skipped} page(s) left for the browser` : ""),
);
if (failures > 0) Deno.exit(1);
