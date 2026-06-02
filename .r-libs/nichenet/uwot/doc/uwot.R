## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(uwot)
library(RSpectra)

## ----plot function------------------------------------------------------------
kabsch <- function(pm, qm) {
  pm_dims <- dim(pm)
  if (!all(dim(qm) == pm_dims)) {
    stop(call. = TRUE, "Point sets must have the same dimensions")
  }
  # The rotation matrix will have (ncol - 1) leading ones in the diagonal
  diag_ones <- rep(1, pm_dims[2] - 1)

  # center the points
  pm <- scale(pm, center = TRUE, scale = FALSE)
  qm <- scale(qm, center = TRUE, scale = FALSE)

  am <- crossprod(pm, qm)

  svd_res <- svd(am)
  # use the sign of the determinant to ensure a right-hand coordinate system
  d <- determinant(tcrossprod(svd_res$v, svd_res$u))$sign
  dm <- diag(c(diag_ones, d))

  # rotation matrix
  um <- svd_res$v %*% tcrossprod(dm, svd_res$u)

  # Rotate and then translate to the original centroid location of qm
  sweep(t(tcrossprod(um, pm)), 2, -attr(qm, "scaled:center"))
}
iris_pca2 <- prcomp(iris[, 1:4])$x[, 1:2]
plot_umap <- function(coords, col = iris$Species, pca = iris_pca2) {
  plot(kabsch(coords, pca), col = col, xlab = "", ylab = "")
}

## ----basic UMAP---------------------------------------------------------------
set.seed(42)
iris_umap <- umap(iris)
plot_umap(iris_umap)

## ----min_dist 0.5-------------------------------------------------------------
set.seed(42)
iris_umap_md05 <- umap(iris, min_dist = 0.3)
plot_umap(iris_umap_md05)

## ----5 neighbors--------------------------------------------------------------
set.seed(42)
iris_umap_nbrs5 <- umap(iris, n_neighbors = 5, min_dist = 0.3)
plot_umap(iris_umap_nbrs5)

## ----100 neighbors------------------------------------------------------------
set.seed(42)
iris_umap_nbrs100 <- umap(iris, n_neighbors = 100, min_dist = 0.3)
plot_umap(iris_umap_nbrs100)

## ----spca init----------------------------------------------------------------
set.seed(42)
iris_umap_spca <-
  umap(iris,
    init = "spca",
    init_sdev = "range",
    min_dist = 0.3
  )
plot_umap(iris_umap_spca)

## ----UMAP with density scaling------------------------------------------------
set.seed(42)
iris_umapds <- umap(iris, min_dist = 0.3, dens_scale = 0.5)
plot_umap(iris_umapds)

## ----create a UMAP model------------------------------------------------------
set.seed(42)

iris_train <- iris[iris$Species %in% c("setosa", "versicolor"), ]
iris_train_umap <-
  umap(iris_train, min_dist = 0.3, ret_model = TRUE)
plot(
  iris_train_umap$embedding,
  col = iris_train$Species,
  xlab = "",
  ylab = "",
  main = "UMAP setosa + versicolor"
)

## ----embed new coordinates----------------------------------------------------
iris_test <- iris[iris$Species == "virginica", ]
set.seed(42)
iris_test_umap <- umap_transform(iris_test, iris_train_umap)
plot(
  rbind(iris_train_umap$embedding, iris_test_umap),
  col = iris$Species,
  xlab = "",
  ylab = "",
  main = "UMAP transform virginica"
)

## ----echo=FALSE, out.width="75%", fig.cap="MNIST UMAP (Python)"---------------
knitr::include_graphics("mnist-py.png")

## ----echo=FALSE, out.width="75%", fig.cap="MNIST UMAP (R)"--------------------
knitr::include_graphics("mnist-r.png")

