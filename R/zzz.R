.onLoad <- function(libname, pkgname) {
  if (!torch::torch_is_installed()) {
    packageStartupMessage("LibTorch is not installed: run torch::install_torch()")
    return(invisible())
  }
  lib_path <- file.path(torch::torch_install_path(), "lib")
  lantern_init_cpp(normalizePath(lib_path, winslash = "/", mustWork = TRUE))
}
