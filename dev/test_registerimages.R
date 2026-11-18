# test_registerimages.R
# Registers two DLPFC Visium images with SIFT and with SuperPoint via img.matching.
# Run from the package folder (C:/rPackage/img.matching).

# install.packages("png")   # once, if missing
library(png)
library(torch)              # loads LibTorch/Lantern first

devtools::load_all(".")     # compiles src/ and loads the package

# --- paths (forward slashes work on Windows and need no escaping) -----------
base       <- "J:/boxFU/Uni/FU Ba. BioInf/11. Sem/Bachelorarbeit"
img1_path  <- file.path(base, "data/Visium/DLPFC_Visium/DLPFC_Visium_Cropped/151507.png")
img2_path  <- file.path(base, "data/Visium/DLPFC_Visium/DLPFC_Visium_Cropped/151508.png")
model_path <- file.path(base, "Code/C++/superpoint_scripted.pt")

stopifnot(file.exists(img1_path), file.exists(img2_path), file.exists(model_path))

# --- make sure Lantern is initialised (no-op if .onLoad already did it) ------
lantern_init_cpp(file.path(torch_install_path(), "lib"))

# --- read PNG -> grayscale numeric matrix ------------------------------------
# readPNG gives values in [0, 1]: a matrix (gray) or an array H x W x C
# (C = 2 gray+alpha, 3 RGB, 4 RGBA). registerimages_cpp wants one channel.
read_gray <- function(path) {
  img <- readPNG(path)
  if (length(dim(img)) == 2) return(img)
  ch <- dim(img)[3]
  if (ch <= 2) return(img[, , 1])
  0.299 * img[, , 1] + 0.587 * img[, , 2] + 0.114 * img[, , 3]   # RGB(A) -> gray
}

img1 <- as_img_matrix(img1_path)
img2 <- as_img_matrix(img2_path)
cat(sprintf("image1: %d x %d   image2: %d x %d (rows x cols)\n",
            nrow(img1), ncol(img1), nrow(img2), ncol(img2)))

# --- run both detectors -------------------------------------------------------
res_sift <- registerimages_cpp(img1, img2, list(detection = "SIFT"))

res_sp <- registerimages_cpp(img1, img2, list(
  detection  = "SuperPoint",
  model_path = model_path
))

# --- results ------------------------------------------------------------------
show_result <- function(name, res) {
  cat("\n=====", name, "=====\n")
  print(unlist(res$stats))
  cat("\naffine (image1 -> image2, 0-based pixels):\n")
  print(res$affine)
}
show_result("SIFT", res_sift)
show_result("SuperPoint", res_sp)

# --- quick visual check: image1 warped onto image2 with the affine -----------
# R's image() draws rows bottom-up, so flip vertically for display.
warp_affine <- function(src, A, out_dim) {
  if (is.null(A)) return(NULL)
  # inverse mapping: for each output pixel (x, y) find the source pixel
  Ainv <- solve(rbind(A, c(0, 0, 1)))
  g  <- expand.grid(x = 0:(out_dim[2] - 1), y = 0:(out_dim[1] - 1))
  sx <- round(Ainv[1, 1] * g$x + Ainv[1, 2] * g$y + Ainv[1, 3])
  sy <- round(Ainv[2, 1] * g$x + Ainv[2, 2] * g$y + Ainv[2, 3])
  ok <- sx >= 0 & sx < ncol(src) & sy >= 0 & sy < nrow(src)
  out <- matrix(NA_real_, out_dim[1], out_dim[2])
  out[cbind(g$y[ok] + 1, g$x[ok] + 1)] <- src[cbind(sy[ok] + 1, sx[ok] + 1)]
  out
}
show_img <- function(m, main) {
  image(t(m[nrow(m):1, ]), col = gray.colors(256), axes = FALSE, main = main, asp = nrow(m) / ncol(m))
}

op <- par(mfrow = c(1, 3), mar = c(1, 1, 2, 1))
show_img(img2, "image2 (target)")
for (r in list(list("SIFT", res_sift), list("SuperPoint", res_sp))) {
  w <- warp_affine(img1, r[[2]]$affine, dim(img2))
  if (is.null(w)) plot.new() else show_img(w, paste("image1 warped -", r[[1]]))
}
par(op)


