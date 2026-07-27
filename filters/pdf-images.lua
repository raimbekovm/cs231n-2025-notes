-- Point the PDF build at the raster sibling of any AVIF figure.
--
-- Figures are stored as AVIF because that is what a reader downloads, but LaTeX has no
-- AVIF reader and the build fails outright on one. `tools/to_avif.py` leaves a sibling
-- LaTeX can read beside every file it converts — a JPEG when re-encoding wins, and the
-- original PNG when it does not, since a scanned line drawing often survives better as
-- PNG than as a JPEG of the same size. So both extensions have to be tried.
--
-- HTML and every other format are untouched.

local function exists(path)
  local f = io.open(path, "r")
  if f then
    f:close()
    return true
  end
  return false
end

-- Image src is relative to the .qmd (../figures/...), so resolve against that too.
local function resolve(src)
  for _, ext in ipairs({ ".jpg", ".png" }) do
    local candidate = src:gsub("%.avif$", ext)
    if exists(candidate) or exists(candidate:gsub("^%.%./", "")) then
      return candidate
    end
  end
  return nil
end

local function swap(image)
  if not image.src:match("%.avif$") then
    return nil
  end

  local sibling = resolve(image.src)
  if not sibling then
    error("no PNG or JPEG sibling for " .. image.src .. " — run tools/to_avif.py, the PDF cannot embed AVIF")
  end

  image.src = sibling
  return image
end

if FORMAT:match("latex") then
  return { { Image = swap } }
end

return {}
