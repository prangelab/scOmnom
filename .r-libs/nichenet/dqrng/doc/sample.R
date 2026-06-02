## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
evaluate <- FALSE
require(bench)

## ----replacement, eval=evaluate-----------------------------------------------
#  library(dqrng)
#  m <- 1e6
#  n <- 1e4
#  bm <- bench::mark(sample.int(m, n, replace = TRUE),
#                    sample.int(1e4*m, n, replace = TRUE),
#                    dqsample.int(m, n, replace = TRUE),
#                    dqsample.int(1e4*m, n, replace = TRUE),
#                    check = FALSE)

## ----echo=FALSE---------------------------------------------------------------
if (evaluate) {
  saveRDS(bm, "data/replacement.RDS")
} else {
  bm <- readRDS("data/replacement.RDS")
}
knitr::kable(bm[, 1:5])

## ----no-replacement-high, eval=evaluate---------------------------------------
#  library(dqrng)
#  m <- 1e6
#  n <- 6e5
#  bm <- bench::mark(sample.int(m, n),
#                    dqsample.int(m, n),
#                    check = FALSE, min_iterations = 50)

## ----echo=FALSE---------------------------------------------------------------
if (evaluate) {
  saveRDS(bm, "data/no-replacement-high.RDS")
} else {
  bm <- readRDS("data/no-replacement-high.RDS")
}
knitr::kable(bm[, 1:5])

## ----no-replacement-medium, eval=evaluate-------------------------------------
#  library(dqrng)
#  m <- 1e6
#  n <- 1e4
#  bm <- bench::mark(sample.int(m, n),
#                    sample.int(m, n, useHash = TRUE),
#                    dqsample.int(m, n),
#                    check = FALSE)

## ----echo=FALSE---------------------------------------------------------------
if (evaluate) {
  saveRDS(bm, "data/no-replacement-medium.RDS")
} else {
  bm <- readRDS("data/no-replacement-medium.RDS")
}
knitr::kable(bm[, 1:5])

## ----no-replacement-low, eval=evaluate----------------------------------------
#  library(dqrng)
#  m <- 1e6
#  n <- 1e2
#  bm <- bench::mark(sample.int(m, n),
#                    sample.int(m, n, useHash = TRUE),
#                    dqsample.int(m, n),
#                    check = FALSE)

## ----echo=FALSE---------------------------------------------------------------
if (evaluate) {
  saveRDS(bm, "data/no-replacement-low.RDS")
} else {
  bm <- readRDS("data/no-replacement-low.RDS")
}
knitr::kable(bm[, 1:5])

## ----no-replacement-long, eval=evaluate---------------------------------------
#  library(dqrng)
#  m <- 1e10
#  n <- 1e5
#  bm <- bench::mark(sample.int(m, n),
#                    dqsample.int(m, n),
#                    check = FALSE)

## ----echo=FALSE---------------------------------------------------------------
if (evaluate) {
  saveRDS(bm, "data/no-replacement-long.RDS")
} else {
  bm <- readRDS("data/no-replacement-long.RDS")
}
knitr::kable(bm[, 1:5])

## ----eval=FALSE---------------------------------------------------------------
#  no_replace_shuffle <- function(m, n) {
#    tmp <- seq_len(m)
#    for (i in seq_len(n))
#      swap(tmp[i], tmp[i + random_int(m-i)])
#    tmp[1:n]
#  }

## ----eval=FALSE---------------------------------------------------------------
#  no_replace_set <- function(m, n) {
#    result <- vector(mode = "...", length = n) # integer or numeric
#    elems <- new(set, m, n) # set object for storing n objects out of m possible values
#    for (i in seq_len(n))
#      while (TRUE) {
#        v = random_int(m)
#        if (elems.insert(v)) {
#          result[i] = v
#          break
#        }
#      }
#    result
#  }

