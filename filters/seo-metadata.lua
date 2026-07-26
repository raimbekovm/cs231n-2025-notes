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
--- "lectures/01-history.html". Pandoc reports the input file, which may be
--- absolute or project-relative depending on how Quarto invoked it.
local function output_path()
  local inputs = PANDOC_STATE.input_files
  if not inputs or not inputs[1] then
    return nil
  end

  local path = inputs[1]:gsub('\\', '/'):gsub('^%./', '')

  local root = quarto.project and quarto.project.directory
  if root then
    root = root:gsub('\\', '/'):gsub('/$', '')
    -- Anchor to the start so a directory named after the project deeper in the
    -- tree cannot be mistaken for the root.
    if path:sub(1, #root + 1) == root .. '/' then
      path = path:sub(#root + 2)
    end
  end

  if path:sub(1, 1) == '/' then
    return nil -- still absolute: the root did not match, so give up rather than guess
  end

  return (path:gsub('%.[^./]+$', '.html'))
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

  local title = meta_string(meta, 'title') or ''
  local description = meta_string(meta, 'description') or ''
  local author = meta_string(meta, 'author')
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
      { '@type', '@id', 'url', 'headline', 'name', 'description', 'author',
        'inLanguage', 'isAccessibleForFree', 'license', 'isPartOf', 'about',
        'keywords' },
      {
        -- TechArticle rather than Chapter: both are accurate, but only the
        -- former is a type search engines surface.
        ['@type'] = 'TechArticle',
        ['@id'] = url .. '#article',
        url = url,
        headline = title,
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
