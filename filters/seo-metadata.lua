--- Emit the per-page metadata Quarto's book format leaves out.
---
--- Quarto writes og:title/description/image for book pages but no og:url and no
--- rel=canonical, and it never emits structured data. The first two matter
--- because the site is reachable at several URLs (with and without the trailing
--- index.html, and through any fork's Pages domain); without a canonical those
--- split each page's ranking signals. The third matters because search and
--- answer engines lift author, licence and part-of-book relations out of
--- JSON-LD rather than inferring them from prose.
---
--- Everything here needs the page's own absolute URL, which a static
--- include-in-header file cannot provide -- hence a filter.

local stringify = pandoc.utils.stringify

--- Read a metadata key, following a dotted path, as a plain string.
local function meta_string(meta, path)
  local node = meta
  for key in path:gmatch '[^.]+' do
    if type(node) ~= 'table' then
      return nil
    end
    node = node[key]
    if node == nil then
      return nil
    end
  end
  local text = stringify(node)
  return text ~= '' and text or nil
end

--- Read a metadata key as a list of plain strings.
local function meta_list(meta, path)
  local node = meta
  for key in path:gmatch '[^.]+' do
    if type(node) ~= 'table' then
      return {}
    end
    node = node[key]
    if node == nil then
      return {}
    end
  end
  local items = {}
  for _, item in ipairs(node) do
    local text = stringify(item)
    if text ~= '' then
      table.insert(items, text)
    end
  end
  return items
end

--- Escape a string for embedding in JSON inside a <script> element.
--- The angle brackets are not required by JSON, but without them a description
--- containing "</script>" would close the element early.
local function json_string(value)
  local escaped = value
    :gsub('\\', '\\\\')
    :gsub('"', '\\"')
    :gsub('[\n\r\t]', ' ')
    :gsub('<', '\\u003c')
    :gsub('>', '\\u003e')
  return '"' .. escaped .. '"'
end

--- Escape a string for an HTML attribute value.
local function attr(value)
  return (value:gsub('&', '&amp;'):gsub('"', '&quot;'):gsub('<', '&lt;'))
end

--- Serialise a Lua value as JSON. Tables carrying an `ordered` field are
--- written as objects in that key order, so the output stays diffable; plain
--- array-like tables become JSON arrays.
local function encode(value)
  if type(value) == 'string' then
    return json_string(value)
  elseif type(value) == 'boolean' or type(value) == 'number' then
    return tostring(value)
  end

  if value.ordered then
    local parts = {}
    for _, key in ipairs(value.ordered) do
      local item = value[key]
      if item ~= nil then
        table.insert(parts, json_string(key) .. ':' .. encode(item))
      end
    end
    return '{' .. table.concat(parts, ',') .. '}'
  end

  local parts = {}
  for _, item in ipairs(value) do
    table.insert(parts, encode(item))
  end
  return '[' .. table.concat(parts, ',') .. ']'
end

--- Build an object that `encode` will write with its keys in the given order.
local function object(keys, values)
  values.ordered = keys
  return values
end

--- The rendered page's path relative to the project root, e.g.
--- "lectures/01-history.html".
---
--- Not derived from PANDOC_STATE.input_files: Quarto preprocesses each document
--- into a temporary copy under the session directory and hands pandoc that, so
--- the name pandoc sees is an opaque temp file. `quarto.doc.input_file` is the
--- real source path, and it shares an exact prefix with the project directory.
local function output_path()
  local root = quarto.project and quarto.project.directory
  local input = quarto.doc.input_file
  if not root or not input then
    return nil
  end

  root = root:gsub('\\', '/'):gsub('/$', '')
  input = input:gsub('\\', '/')

  -- Anchor to the start so a directory named after the project deeper in the
  -- tree cannot be mistaken for the root.
  if input:sub(1, #root + 1) ~= root .. '/' then
    return nil -- give up rather than guess at a wrong canonical URL
  end

  return (input:sub(#root + 2):gsub('%.[^./]+$', '.html'))
end

function Pandoc(doc)
  if not FORMAT:match 'html' then
    return nil
  end

  local meta = doc.meta

  -- Which of these Quarto actually hands to a filter depends on the project
  -- type, so try the native spellings first and fall back to the copy under
  -- `seo:`, which is ours and therefore always present.
  local site = meta_string(meta, 'site-url')
    or meta_string(meta, 'book.site-url')
    or meta_string(meta, 'website.site-url')
    or meta_string(meta, 'seo.site-url')
  local path = output_path()

  if not site or not path then
    quarto.log.warning(
      'seo-metadata: no canonical URL emitted (site-url=' ..
      tostring(site) .. ', path=' .. tostring(path) .. ')')
    return nil
  end

  local base = site:gsub('/$', '') .. '/'
  local url = base .. path

  -- Book numbering has already been folded into the title by the time filters
  -- run, so a chapter arrives as "2\u{a0} History of Computer Vision". A heading
  -- in structured data should read as a name, not as a position in this book,
  -- and the position is carried by the breadcrumb anyway.
  --
  -- The non-breaking space is what makes this safe to strip: it is the
  -- separator Quarto inserts and never something an author would type, so a
  -- title that legitimately opens with a number survives untouched. Lua
  -- patterns match bytes, so U+00A0 has to be spelled out as its UTF-8 pair
  -- rather than left to %s.
  local title = (meta_string(meta, 'title') or '')
    :gsub('^[%d%.]+\194\160%s*', '')
  local subtitle = meta_string(meta, 'subtitle')
  local description = meta_string(meta, 'description') or ''
  -- Chapters do not inherit the book's author into their own metadata.
  local author = meta_string(meta, 'author') or meta_string(meta, 'book.author')
  local author_url = meta_string(meta, 'seo.author-url')
  local license = meta_string(meta, 'seo.license-url')
  local keywords = meta_list(meta, 'seo.keywords')

  local book_title = meta_string(meta, 'book.title') or title
  local book_id = base .. '#book'

  local person = author and object(
    { '@type', 'name', 'url' },
    { ['@type'] = 'Person', name = author, url = author_url })

  local subjects = {}
  for _, keyword in ipairs(keywords) do
    table.insert(subjects, object(
      { '@type', 'name' },
      { ['@type'] = 'Thing', name = keyword }))
  end

  -- The landing page carries the Book node that every chapter points back to;
  -- chapters carry an article and a breadcrumb. Both go in one @graph so the
  -- page emits a single script element.
  local graph = {}

  if path == 'index.html' then
    table.insert(graph, object(
      { '@type', '@id', 'url', 'name', 'description', 'author', 'inLanguage',
        'isAccessibleForFree', 'license', 'bookFormat', 'genre', 'about' },
      {
        ['@type'] = 'Book',
        ['@id'] = book_id,
        url = base,
        name = book_title,
        description = description ~= '' and description or nil,
        author = person,
        inLanguage = 'en',
        isAccessibleForFree = true,
        license = license,
        bookFormat = 'https://schema.org/EBook',
        genre = 'Lecture notes',
        about = #subjects > 0 and subjects or nil,
      }))
    table.insert(graph, object(
      { '@type', '@id', 'url', 'name', 'inLanguage' },
      {
        ['@type'] = 'WebSite',
        ['@id'] = base .. '#website',
        url = base,
        name = book_title,
        inLanguage = 'en',
      }))
  else
    table.insert(graph, object(
      { '@type', '@id', 'url', 'headline', 'alternativeHeadline', 'name',
        'description', 'author', 'inLanguage', 'isAccessibleForFree', 'license',
        'isPartOf', 'about', 'keywords' },
      {
        -- TechArticle rather than Chapter: both are accurate, but only the
        -- former is a type search engines surface.
        ['@type'] = 'TechArticle',
        ['@id'] = url .. '#article',
        url = url,
        headline = title,
        alternativeHeadline = subtitle,
        name = title,
        description = description ~= '' and description or nil,
        author = person,
        inLanguage = 'en',
        isAccessibleForFree = true,
        license = license,
        isPartOf = object({ '@type', '@id', 'name', 'url' }, {
          ['@type'] = 'Book',
          ['@id'] = book_id,
          name = book_title,
          url = base,
        }),
        about = #subjects > 0 and subjects or nil,
        keywords = #keywords > 0 and table.concat(keywords, ', ') or nil,
      }))
    table.insert(graph, object(
      { '@type', 'itemListElement' },
      {
        ['@type'] = 'BreadcrumbList',
        itemListElement = {
          object({ '@type', 'position', 'name', 'item' }, {
            ['@type'] = 'ListItem',
            position = 1,
            name = book_title,
            item = base,
          }),
          object({ '@type', 'position', 'name', 'item' }, {
            ['@type'] = 'ListItem',
            position = 2,
            name = title,
            item = url,
          }),
        },
      }))
  end

  local jsonld = encode(object({ '@context', '@graph' }, {
    ['@context'] = 'https://schema.org',
    ['@graph'] = graph,
  }))

  quarto.doc.include_text('in-header', table.concat({
    '<link rel="canonical" href="' .. attr(url) .. '">',
    '<meta property="og:url" content="' .. attr(url) .. '">',
    '<script type="application/ld+json">' .. jsonld .. '</script>',
  }, '\n'))

  return nil
end
