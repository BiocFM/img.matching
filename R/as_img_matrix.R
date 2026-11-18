#' @include AllGenerics.R
NULL

setMethod("as_img_matrix", "matrix", function(x, ...) {
  x
})

setMethod("as_img_matrix", "Image", function(x, channel = 1, ...) {
  d <- EBImage::imageData(x)
  if (length(dim(d)) == 3) d <- d[, , channel]       #reduce dimensions so its grayscale
  t(d)
})

setOldClass("magick-image")
setMethod("as_img_matrix", "magick-image", function(x, ...) {
  arr <- as.integer(magick::image_data(x, channels = "gray"))
  arr[, , 1]
})

setMethod("as_img_matrix", "character", function(x, channel = 1, ...) {
  if (!file.exists(x)) stop("File not found: ", x, call. = FALSE)
  ext <- tolower(x)
  if (grepl("\\.tiff?$", ext)) {
    # OME-TIFF: one page per channel
    arr <- ijtiff::read_tif(x, frames = channel, msg = FALSE)
    arr[, , 1, 1]
  } else if (grepl("\\.png$", ext)) {
    arr <- png::readPNG(x)
    if (length(dim(arr)) == 3) {
      if (dim(arr)[3] >= 3) {
        # RGB / RGBA -> luminance (same weights as OpenCV's COLOR_BGR2GRAY), alpha ignored
        arr <- 0.299 * arr[, , 1] + 0.587 * arr[, , 2] + 0.114 * arr[, , 3]
      } else {
        arr <- arr[, , 1]   # gray + alpha -> gray
      }
    }
    arr
  } else {
    stop("Unsupported file type: ", basename(x), call. = FALSE)
  }
})
