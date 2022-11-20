cc_library(
    name = "mimalloc",
    srcs = ["src/static.c"],
    textual_hdrs = glob(["src/*.c", "src/*.h"], exclude = ["src/static.c"]),
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)