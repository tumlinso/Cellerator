args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
x <- Matrix::sparseMatrix(
    i = c(1L, 4L, 2L, 3L), j = c(1L, 1L, 3L, 4L),
    x = c(1.25, 0, 2.5, -7), dims = c(4L, 5L),
    dimnames = list(c("gene-a", "gene-b", NA_character_, ""),
                    c("cell-a", "cell-b", "cell-c", "cell-d", "cell-e")))
stopifnot(inherits(x, "dgCMatrix"), identical(x@p, c(0L, 2L, 2L, 3L, 4L, 4L)))
saveRDS(x, args[[1L]], version = 3L, compress = "gzip")
