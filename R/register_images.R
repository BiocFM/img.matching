#' Register two images
#'
#' Detects keypoints (SIFT or SuperPoint), matches them with FLANN and
#' Lowe's ratio test, and estimates a transform mapping `moving` onto `fixed`.
#'
#' @param moving,fixed (image1 and image 2).
#' @param detector Keypoint detector: `"sift"` or `"superpoint"`.
#' @param max_keypoints
#' ...
#' @return An `img_registration` object.
#' @export

register_images <- function(moving, fixed,
                            detector = c("sift", "superpoint"),
                            model_path = "C:/rPackage/img.matching/inst/models/superpoint.pt",
                            max_keypoints = 20000){

  imgm <- as_img_matrix(moving)
  imgf <- as_img_matrix(fixed)

  show_result <- function(name, res) {
    cat("\n=====", name, "=====\n")
    print(unlist(res$stats))
    cat("\naffine (image1 -> image2, 0-based pixels):\n")
    print(res$affine)
  }

  if (detector == "sift"){
  res_sift <- registerimages_cpp(imgm, imgf, list(detection = "SIFT"))
  show_result("SIFT", res_sift)
  }
  else if(detector == "superpoint"){
  res_sp <- registerimages_cpp(imgm, imgf, list(
    detection  = "SuperPoint",
    model_path = model_path))
  show_result("SuperPoint", res_sp)
  }
  else{
    print("Detector is invalid. Choose either superpoint or sift")
  }


}
