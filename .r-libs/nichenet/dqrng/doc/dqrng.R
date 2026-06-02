## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
N <- 1e7
piR <- function(n, rng = stats::runif) {
    x <- rng(n)
    y <- rng(n)
    4 * sum(sqrt(x^2 + y^2) < 1.0) / n
}
set.seed(42)
system.time(cat("pi ~= ", piR(N), "\n"))

## -----------------------------------------------------------------------------
library(dqrng)
dqset.seed(42)
system.time(cat("pi ~= ", piR(N, rng = dqrng::dqrunif), "\n"))

## -----------------------------------------------------------------------------
system.time(stats::runif(N))
system.time(dqrng::dqrunif(N))

## -----------------------------------------------------------------------------
system.time(stats::rexp(N))
system.time(dqrng::dqrexp(N))

## -----------------------------------------------------------------------------
system.time(stats::rnorm(N))
system.time(dqrng::dqrnorm(N))

## -----------------------------------------------------------------------------
system.time(for (i in 1:100)   sample.int(N, N/100, replace = TRUE))
system.time(for (i in 1:100) dqsample.int(N, N/100, replace = TRUE))
system.time(for (i in 1:100)   sample.int(N, N/100))
system.time(for (i in 1:100) dqsample.int(N, N/100))

## ----register-----------------------------------------------------------------
register_methods()
set.seed(4711);   stats::runif(5)
set.seed(4711);   dqrng::dqrunif(5)
dqset.seed(4711); stats::rnorm(5)
dqset.seed(4711); dqrng::dqrnorm(5)
set.seed(4711);   stats::rt(5, 10)
dqset.seed(4711); stats::rt(5, 10)
set.seed(4711);   stats::rexp(5, 10)
set.seed(4711);   dqrng::dqrexp(5, 10)
restore_methods()

## ----state--------------------------------------------------------------------
(state <- dqrng_get_state())
dqrunif(5)
# many other operations, that could even change the used RNG type
dqrng_set_state(state)
dqrunif(5)

