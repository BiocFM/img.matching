// LibTorch from Rcpp, compiled by Rtools -- via R torch's Lantern C API.

// Include lantern.h in exactly ONE .cpp of the package (this one): without
// LANTERN_HEADERS_ONLY it defines the pointers and lanternInit().

#include <lantern/lantern.h>   
#include <Rcpp.h>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Glue that lantern.h expects the host package to provide
// ---------------------------------------------------------------------------
int  lanternLogEnabled = 0;
bool lantern_loaded    = false;

void check_lantern_loaded()
{
    if (!lantern_loaded)
        Rcpp::stop("Lantern is not initialised. Call lantern_init_cpp() first (done in .onLoad).");
}

// Called after every lantern_* call: turn LibTorch errors into R errors
void lantern_host_handler()
{
    const char* err = lanternLastError();
    if (err != NULL) {
        std::string msg(err);
        lanternLastErrorClear();
        Rcpp::stop("LibTorch error: " + msg);
    }
}

// ---------------------------------------------------------------------------
// Initialisation (called once from R's .onLoad)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
void lantern_init_cpp(std::string lib_path)
{
    if (lantern_loaded) return;
    std::string error;
    if (!lanternInit(lib_path, &error))
        Rcpp::stop("Could not load lantern from '" + lib_path + "': " + error);
}

