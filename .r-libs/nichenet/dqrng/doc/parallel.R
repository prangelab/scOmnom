## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
library(dqrng)
evaluate <- FALSE

## ----eval=evaluate------------------------------------------------------------
#  library(parallel)
#  cl <- parallel::makeCluster(2)
#  res <- clusterApply(cl, 1:8, function(stream, seed, N) {
#    library(dqrng)
#    dqRNGkind("Threefry")
#    dqset.seed(seed, stream)
#    dqrnorm(N)
#  }, 42, 1e6)
#  stopCluster(cl)
#  
#  res <- matrix(unlist(res), ncol = 8)
#  symnum(x = cor(res), cutpoints = c(0.001, 0.003, 0.999),
#         symbols = c(" ", "?", "!", "1"),
#         abbr.colnames = FALSE, corr = TRUE)

## ----eval=evaluate------------------------------------------------------------
#  dqset.seed(42); norm1 <- rnorm_para(22, threads = 1)
#  dqset.seed(42); norm2 <- rnorm_para(22, threads = 4)
#  identical(norm1, norm2)
#  #> [1] TRUE

## ----eval=evaluate------------------------------------------------------------
#  n <- 1e6
#  bench::mark(stats::runif(n),
#              dqrng::dqrunif(n),
#              runif_para(n, threads = 2L),
#              runif_para(n, threads = 1L),
#              check = FALSE)[, 1:6]
#  bench::mark(stats::rnorm(n),
#              dqrng::dqrnorm(n),
#              rnorm_para(n, threads = 2L),
#              rnorm_para(n, threads = 1L),
#              check = FALSE)[, 1:6]
#  bench::mark(stats::rexp(n),
#              dqrng::dqrexp(n),
#              rexp_para(n, threads = 2L),
#              rexp_para(n, threads = 1L),
#              check = FALSE)[, 1:6]

## ----eval=evaluate------------------------------------------------------------
#  dqset.seed(153)
#  runif_para(30)
#  #>   [1] 0.87693642 0.14323366 0.33129746 0.07856319 0.80991119 0.37524485
#  #>  [7] 0.90387542 0.38746776 0.30473153 0.01102334 0.21272306 0.11975609
#  #> [13] 0.98440547 0.13373340 0.82823735 0.87196225 0.14920422 0.27723804
#  #> [19] 0.59308120 0.07853078 0.63040483 0.21707435 0.25876379 0.81296194
#  #> [25] 0.53645030 0.29976254 0.37159454 0.38683266 0.03737063 0.03359113
#  runif_para(30) # Different values, as expected
#  #>  [1] 0.90407135 0.73543499 0.09026296 0.90321975 0.66162669 0.51716146
#  #>  [7] 0.74186366 0.41125413 0.17581202 0.68547734 0.11766549 0.82316789
#  #> [13] 0.40565668 0.44854610 0.95477820 0.64388593 0.31991691 0.02239872
#  #> [19] 0.13687388 0.32476719 0.67093851 0.05564081 0.76817620 0.49502455
#  #> [25] 0.07459706 0.25493312 0.14019729 0.89704659 0.40548199 0.53800443

## ----eval=evaluate------------------------------------------------------------
#  # Seed used in the first thread
#  dqset.seed(546, 1); (v1 <- rnorm_para(8, streams = 4))
#  #> [1]  0.01904358  0.57750157  0.39156879 -1.72594164  1.24949453 -0.87535133
#  #> [7] -0.49878776  0.26077249
#  # Seed used in the second thread
#  dqset.seed(546, 2); (v2 <- rnorm_para(8, streams = 4))
#  #> [1]  0.3915688 -1.7259416  1.2494945 -0.8753513 -0.4987878  0.2607725
#  #> [7]  1.2018189 -0.1060487

## ----eval=evaluate------------------------------------------------------------
#  # Seed used in the first thread
#  dqset.seed(546, 1); (v1 <- rnorm_para(8, streams = 4))
#  #> [1]  0.01904358  0.57750157  0.39156879 -1.72594164  1.24949453 -0.87535133
#  #> [7] -0.49878776  0.26077249
#  # Seed used in the second thread
#  dqset.seed(546, 5); (v2 <- rnorm_para(8, streams = 4))
#  #> [1]  1.2018189 -0.1060487 -0.8532641  0.6531933 -0.8304053 -0.4745548
#  #> [7] -0.4211618 -0.5871540

