args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Expected a JSON config path.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(purrr)
  library(readr)
  library(tibble)
  library(nichenetr)
})

cfg <- jsonlite::fromJSON(args[[1]], simplifyVector = TRUE)
options(timeout = max(600L, getOption("timeout", 60L)))

read_lines_clean <- function(path) {
  vals <- readLines(path, warn = FALSE)
  vals <- trimws(vals)
  vals <- vals[nzchar(vals)]
  unique(vals)
}

load_model_urls <- function(organism) {
  if (tolower(organism) != "human") {
    stop("scOmnom NicheNet v1 currently supports organism='human' only.", call. = FALSE)
  }
  list(
    ligand_target_matrix = "https://zenodo.org/record/7074291/files/ligand_target_matrix_nsga2r_final.rds",
    lr_network = "https://zenodo.org/record/7074291/files/lr_network_human_21122021.rds",
    weighted_networks = "https://zenodo.org/record/7074291/files/weighted_networks_nsga2r_final.rds"
  )
}

urls <- load_model_urls(cfg$organism)
ligand_target_matrix <- readRDS(url(urls$ligand_target_matrix))
lr_network <- readRDS(url(urls$lr_network))
weighted_networks <- readRDS(url(urls$weighted_networks))

sender_expressed_genes <- read_lines_clean(cfg$sender_expressed_genes_file)
receiver_expressed_genes <- read_lines_clean(cfg$receiver_expressed_genes_file)
geneset_oi <- read_lines_clean(cfg$geneset_file)
background_genes <- read_lines_clean(cfg$background_genes_file)

potential_ligands <- lr_network %>%
  filter(from %in% sender_expressed_genes & to %in% receiver_expressed_genes) %>%
  distinct(from) %>%
  rename(ligand = from)

if (tolower(cfg$signal_scope) == "secreted" && "database" %in% colnames(lr_network)) {
  potential_ligands <- lr_network %>%
    filter(from %in% sender_expressed_genes & to %in% receiver_expressed_genes) %>%
    filter(grepl("secret", database, ignore.case = TRUE)) %>%
    distinct(from) %>%
    rename(ligand = from)
}

if (nrow(potential_ligands) == 0) {
  stop("No potential ligands remained after sender/receiver filtering.", call. = FALSE)
}

ligand_activities <- predict_ligand_activities(
  geneset = geneset_oi,
  background_expressed_genes = background_genes,
  ligand_target_matrix = ligand_target_matrix,
  potential_ligands = potential_ligands$ligand,
  single = TRUE
)
ligand_activities <- as_tibble(ligand_activities) %>%
  arrange(desc(pearson))

best_upstream_ligands <- ligand_activities %>%
  pull(test_ligand) %>%
  head(as.integer(cfg$top_n_ligands))

ligand_target_links_df <- best_upstream_ligands %>%
  lapply(
    get_weighted_ligand_target_links,
    geneset = geneset_oi,
    ligand_target_matrix = ligand_target_matrix,
    n = as.integer(cfg$top_n_targets)
  ) %>%
  bind_rows()

ligand_receptor_links_df <- get_weighted_ligand_receptor_links(
  best_upstream_ligands,
  receiver_expressed_genes,
  lr_network,
  weighted_networks$lr_sig
)

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
write_tsv(ligand_activities, file.path(cfg$output_dir, "ligand_activity.tsv"))
write_tsv(ligand_target_links_df, file.path(cfg$output_dir, "ligand_target_links.tsv"))
write_tsv(as_tibble(ligand_receptor_links_df), file.path(cfg$output_dir, "ligand_receptor_links.tsv"))
write_tsv(potential_ligands, file.path(cfg$output_dir, "potential_ligands.tsv"))
