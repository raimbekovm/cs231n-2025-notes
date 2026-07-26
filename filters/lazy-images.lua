--- Defer off-screen images and reserve their space ahead of time.
---
--- A lecture page carries 15-30 figures and several megabytes of them, all of
--- which the browser would otherwise fetch before the reader has scrolled past
--- the first screen. Quarto applies `loading="lazy"` only to listings, so body
--- images need this filter.
---
--- Lazy loading on its own causes layout shift, because a deferred image has no
--- height until it arrives. Rather than setting width/height attributes -- which
--- Quarto also uses for its own figure sizing, often as percentages -- this sets
--- a CSS `aspect-ratio` from the file's intrinsic dimensions. The browser then
--- reserves the right box while leaving Quarto's sizing untouched.

local pandoc_image = require 'pandoc.image'

-- The first image is usually above the fold; deferring it would delay the
-- largest contentful paint rather than help it.
local is_first = true

-- One CSS rule per measured image, emitted together at the end of the document.
local ratio_rules = {}
local ratio_count = 0

-- Directory of the file being rendered. Image paths in a .qmd are written
-- relative to that file, but pandoc's working directory is the project root,
-- so they need rebasing before they can be opened.
local function input_dir()
  local inputs = PANDOC_STATE.input_files
  if not inputs or not inputs[1] then
    return nil
  end
  return inputs[1]:match '^(.*)[/\\][^/\\]*$'
end

local base_dir = input_dir()

--- Read a file, returning its contents or nil.
local function read_file(path)
  local file = io.open(path, 'rb')
  if not file then
    return nil
  end
  local data = file:read 'a'
  file:close()
  return data
end

--- Intrinsic dimensions of a local image, or nil if they can't be determined.
local function intrinsic_size(src)
  if src:match '^%a[%w+.-]*:' then
    return nil -- remote image; nothing to measure
  end

  local data = read_file(src)
  if not data and base_dir then
    data = read_file(base_dir .. '/' .. src)
  end
  if not data then
    return nil
  end

  local ok, size = pcall(pandoc_image.size, data)
  if not ok or type(size) ~= 'table' then
    return nil
  end
  local w, h = tonumber(size.width), tonumber(size.height)
  if not w or not h or w <= 0 or h <= 0 then
    return nil
  end
  return w, h
end

function Image(img)
  -- These attributes mean nothing outside HTML.
  if not FORMAT:match 'html' then
    return nil
  end

  if is_first then
    is_first = false
    return nil
  end

  img.attributes.loading = 'lazy'
  img.attributes.decoding = 'async'

  -- The ratio cannot go in the image's own style attribute (Quarto hoists that
  -- onto the enclosing float div, constraining the caption too), nor in
  -- width/height attributes (Quarto already uses those for its percentage
  -- sizing, and overwriting them would pin the image to its pixel size). So
  -- tag the image with a generated class and collect a stylesheet rule for it.
  local width, height = intrinsic_size(img.src)
  if width then
    ratio_count = ratio_count + 1
    local class = 'fig-ar-' .. ratio_count
    img.classes:insert(class)
    table.insert(ratio_rules,
      string.format('img.%s{aspect-ratio:%d/%d}', class, width, height))
  end

  return img
end

--- Emit the collected aspect-ratio rules once, at the end of the document.
function Pandoc(doc)
  if #ratio_rules == 0 then
    return doc
  end
  doc.blocks:insert(pandoc.RawBlock('html',
    '<style>' .. table.concat(ratio_rules) .. '</style>'))
  return doc
end
