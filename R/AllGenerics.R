#' Convert an image to a numeric matrix
#'
#' Internal converter used by [register_images()].
#' @param x An image: matrix, EBImage `Image`, magick image, or file path.
#' @param ... Passed to methods (e.g. `channel`).
#' @keywords internal

setGeneric("as_img_matrix", function(x, ...) standardGeneric("as_img_matrix"))
