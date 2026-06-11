import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from scomnom.cli import app

runner = CliRunner()


# ---------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------
def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scOmnom CLI" in result.output


# ---------------------------------------------------------
# cell-qc
# ---------------------------------------------------------
def test_cell_qc_help():
    result = runner.invoke(app, ["cell-qc", "--help"])
    assert result.exit_code == 0
    assert "Generate QC comparisons" in result.output


def test_cell_qc_requires_any_input():
    result = runner.invoke(app, ["cell-qc", "--out", "outdir"])
    # All inputs missing
    assert result.exit_code != 0
    assert "Provide at least one input" in result.output


@patch("scomnom.cli.run_cell_qc")
def test_cell_qc_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "cell-qc",
            "--raw", "raw_dir",
            "--out", "outdir"
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.raw_sample_dir == "raw_dir"
    assert cfg.output_dir.name == "outdir"


# ---------------------------------------------------------
# load-and-filter
# ---------------------------------------------------------
def test_load_and_filter_help():
    result = runner.invoke(app, ["load-and-filter", "--help"])
    assert result.exit_code == 0
    assert "preprocessing pipeline" in result.output


def test_load_and_filter_requires_output_and_metadata():
    result = runner.invoke(app, ["load-and-filter"])
    assert result.exit_code != 0
    assert "Output directory (required)" in result.output


def test_load_and_filter_mutually_exclusive_raw_filtered():
    result = runner.invoke(
        app,
        [
            "load-and-filter",
            "--raw-sample-dir", "rawdir",
            "--filtered-sample-dir", "filtered",
            "--metadata-tsv", "meta.tsv",
            "--out", "out"
        ]
    )
    assert result.exit_code != 0
    assert "Cannot specify both" in result.output


@patch("scomnom.cli.run_load_and_filter")
def test_load_and_filter_dispatch(mock_run, tmp_path):
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample\tcol\nA\t1\n")

    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "load-and-filter",
            "--raw-sample-dir", "raw",
            "--metadata-tsv", str(meta),
            "--out", str(out),
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.raw_sample_dir == "raw"
    assert cfg.metadata_tsv == meta


# ---------------------------------------------------------
# integrate
# ---------------------------------------------------------
def test_integrate_help():
    result = runner.invoke(app, ["integrate", "--help"])
    assert result.exit_code == 0
    assert "batch correction" in result.output


def test_integrate_requires_input():
    result = runner.invoke(app, ["integrate"])
    assert result.exit_code != 0
    assert "Input h5ad" in result.output


def test_integrate_invalid_method_rejected():
    result = runner.invoke(
        app,
        [
            "integrate",
            "--input-path", "adata.h5ad",
            "--methods", "BadTool"
        ]
    )
    assert result.exit_code != 0
    assert "Invalid method" in result.output


@patch("scomnom.cli.run_integration")
def test_integrate_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "integrate",
            "--input-path", "adata.h5ad",
            "--methods", "Scanorama",
            "--methods", "Harmony",
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.methods == ["Scanorama", "Harmony"]
    assert cfg.input_path == "adata.h5ad"


# ---------------------------------------------------------
# cluster-and-annotate
# ---------------------------------------------------------
def test_cluster_help():
    result = runner.invoke(app, ["cluster-and-annotate", "--help"])
    assert result.exit_code == 0
    assert "Perform clustering" in result.output


def test_cluster_requires_input():
    # Missing --input, but not using --list-models or --download-models
    result = runner.invoke(app, ["cluster-and-annotate"])
    assert result.exit_code != 0
    assert "Missing required option --input" in result.output


# --- list models ----------------------------------------------------------
@patch("scomnom.cli.get_available_celltypist_models", return_value=[{"name": "Test.pkl"}])
def test_cluster_list_models(mock_models):
    result = runner.invoke(app, ["cluster-and-annotate", "--list-models"])
    assert result.exit_code == 0
    assert "Available CellTypist models" in result.output
    mock_models.assert_called_once()


# --- download models ------------------------------------------------------
@patch("scomnom.cli.download_all_celltypist_models")
def test_cluster_download_models(mock_dl):
    result = runner.invoke(app, ["cluster-and-annotate", "--download-models"])
    assert result.exit_code == 0
    mock_dl.assert_called_once()


# --- dispatch full clustering ---------------------------------------------
@patch("scomnom.cli.run_clustering")
def test_cluster_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "cluster-and-annotate",
            "--input-path", "integrated.h5ad",
            "--output-dir", "outdir",
            "--output-name", "adata.clustered.annotated.test",
            "--res-min", "0.2",
            "--res-max", "1.2",
        ]
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.input_path == "integrated.h5ad"
    assert cfg.output_dir == "outdir"
    assert cfg.output_name == "adata.clustered.annotated.test"
    assert cfg.res_min == 0.2
    assert cfg.res_max == 1.2


@patch("scomnom.cli.run_clustering")
def test_cluster_dispatch_force_celltypist_recompute(mock_run):
    result = runner.invoke(
        app,
        [
            "cluster-and-annotate",
            "--input-path", "integrated.h5ad",
            "--force-celltypist-recompute",
        ]
    )
    assert result.exit_code == 0
    cfg = mock_run.call_args[0][0]
    assert cfg.force_celltypist_recompute is True


@patch("scomnom.cli.run_clustering")
def test_cluster_dispatch_celltypist_model_none_string_disables(mock_run):
    result = runner.invoke(
        app,
        [
            "cluster-and-annotate",
            "--input-path", "integrated.h5ad",
            "--celltypist-model", "None",
        ]
    )
    assert result.exit_code == 0
    cfg = mock_run.call_args[0][0]
    assert cfg.celltypist_model is None


# ---------------------------------------------------------
# adata-ops
# ---------------------------------------------------------
def test_adata_ops_help():
    result = runner.invoke(app, ["adata-ops", "--help"])
    assert result.exit_code == 0
    assert "AnnData object operations" in result.output
    assert "rename" in result.output
    assert "subset" in result.output
    assert "annotation-merge" in result.output
    assert "metadata-import" in result.output


def test_markers_and_de_help_includes_enrichment():
    result = runner.invoke(app, ["markers-and-de", "--help"])
    assert result.exit_code == 0
    assert "enrichment" in result.output
    assert "ccc" in result.output


def test_adata_ops_rename_requires_mapping():
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "rename",
            "--input-path", "adata.h5ad",
        ],
    )
    assert result.exit_code != 0
    assert "rename-idents-file" in result.output


def test_adata_ops_subset_requires_mapping():
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "subset",
            "--input-path", "adata.h5ad",
        ],
    )
    assert result.exit_code != 0
    assert "subset-mapping-tsv" in result.output


@patch("scomnom.cli.run_adata_ops")
def test_adata_ops_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "subset",
            "--input-path", "adata.h5ad",
            "--subset-mapping-tsv", "mapping.tsv",
            "--output-format", "zarr",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.input_path == "adata.h5ad"
    assert cfg.subset_mapping_tsv == "mapping.tsv"
    assert cfg.output_format == "zarr"


@patch("scomnom.cli.run_adata_ops")
def test_adata_ops_rename_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "rename",
            "--input-path", "adata.h5ad",
            "--rename-idents-file", "mapping.tsv",
            "--output-name", "adata.renamed",
            "--rename-round-name", "refined_idents",
            "--collapse-same-labels",
            "--update-existing-round",
            "--target-round-id", "r5_refined_idents",
            "--no-set-active",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert str(cfg.input_path) == "adata.h5ad"
    assert cfg.operation == "rename"
    assert str(cfg.rename_idents_file) == "mapping.tsv"
    assert cfg.output_name == "adata.renamed"
    assert cfg.rename_round_name == "refined_idents"
    assert cfg.rename_collapse_same_labels is True
    assert cfg.update_existing_round is True
    assert cfg.target_round_id == "r5_refined_idents"
    assert cfg.rename_set_active is False


@patch("scomnom.cli.run_adata_ops")
def test_adata_ops_annotation_merge_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "annotation-merge",
            "--input-path", "adata.h5ad",
            "--child-path", "child1.h5ad",
            "--child-path", "child2.h5ad",
            "--update-existing-round",
            "--target-round-id", "r4_subset_annotation",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert str(cfg.input_path) == "adata.h5ad"
    assert cfg.operation == "annotation_merge"
    assert tuple(map(str, cfg.child_paths)) == ("child1.h5ad", "child2.h5ad")
    assert cfg.update_existing_round is True
    assert cfg.target_round_id == "r4_subset_annotation"


@patch("scomnom.cli.run_adata_ops")
def test_adata_ops_metadata_import_dispatch(mock_run):
    result = runner.invoke(
        app,
        [
            "adata-ops",
            "metadata-import",
            "--input-path", "adata.h5ad",
            "--metadata-file", "meta.tsv",
            "--metadata-key", "sample_id",
            "--obs-key", "sample_id",
            "--column", "condition",
            "--column", "patient",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert str(cfg.input_path) == "adata.h5ad"
    assert cfg.operation == "metadata_import"
    assert str(cfg.metadata_file) == "meta.tsv"
    assert cfg.metadata_key == "sample_id"
    assert cfg.obs_key == "sample_id"
    assert cfg.metadata_columns == ("condition", "patient")
    assert cfg.output_dir is None
    assert str(cfg.resolved_output_dir).endswith("/results")


@patch("scomnom.cli.run_cluster_vs_rest")
def test_markers_default_output_name_includes_round_id(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "markers",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r4_subset_annotation",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.markers_r4_subset_annotation"
    assert cfg.output_dir == Path("results")


@patch("scomnom.cli.run_within_cluster")
def test_de_default_output_name_includes_round_id(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "de",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r5_archetypes",
            "--condition-keys", "timepoint",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.de_r5_archetypes"
    assert cfg.output_dir == Path("results")


@patch("scomnom.cli.run_within_cluster")
def test_de_default_output_dir_uses_input_results_parent(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "de",
            "--input-path", "results/adata.zarr.tar.zst",
            "--condition-keys", "timepoint",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_dir == Path("results")


@patch("scomnom.cli.run_composition")
def test_da_default_output_name_includes_round_id(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "da",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r6_myawesomecustomround",
            "--condition-keys", "timepoint",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.da_r6_myawesomecustomround"


@patch("scomnom.cli.run_enrichment_cluster")
def test_enrichment_cluster_default_output_name_includes_round_id(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "cluster",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r5_archetypes",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.enrichment_r5_archetypes"


@patch("scomnom.cli.run_liana_ccc")
def test_ccc_liana_default_output_name_includes_round_id(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "liana",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r5_archetypes",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.ccc_liana_r5_archetypes"


@patch("scomnom.cli.run_liana_ccc")
def test_ccc_liana_method_and_resource_propagate_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "liana",
            "--input-path", "adata.zarr.tar.zst",
            "--condition-key", "sex",
            "--condition-value", "female,male",
            "--compare-level", "female",
            "--compare-level", "male",
            "--dataset-key", "tissue",
            "--source-level", "liver",
            "--target-level", "fat",
            "--signal-scope", "secreted",
            "--liana-method", "rank_aggregate",
            "--liana-method", "cellphonedb",
            "--liana-method", "natmi",
            "--resource", "CellPhoneDB",
            "--expr-prop", "0.2",
            "--input-mode", "lognorm",
            "--lognorm-target-sum", "5000",
            "--no-use-raw",
            "--layer", "lognorm",
            "--n-perms", "0",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.ccc_condition_key == "sex"
    assert cfg.ccc_condition_keys == ("sex",)
    assert cfg.ccc_condition_values == ("female", "male")
    assert cfg.ccc_compare_levels == ("female", "male")
    assert cfg.ccc_dataset_key == "tissue"
    assert cfg.ccc_source_levels == ("liver",)
    assert cfg.ccc_target_levels == ("fat",)
    assert cfg.ccc_signal_scope == "secreted"
    assert cfg.liana_methods == ("rank_aggregate", "cellphonedb", "natmi")
    assert cfg.liana_resource == "cellphonedb"
    assert cfg.liana_expr_prop == pytest.approx(0.2)
    assert cfg.liana_input_mode == "lognorm"
    assert cfg.liana_lognorm_target_sum == pytest.approx(5000.0)


@patch("scomnom.cli.run_liana_paired_rescore")
def test_ccc_liana_paired_config_propagates(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "liana-paired",
            "--input-path", "adata.zarr.tar.zst",
            "--candidate-events", "liana_rank_aggregate.tsv",
            "--condition-key", "sex@MASLD",
            "--condition-value", "yes",
            "--compare-level", "female",
            "--compare-level", "male",
            "--dataset-key", "tissue",
            "--source-level", "vfat",
            "--target-level", "liv",
            "--pairing-key", "sample_id",
            "--input-mode", "lognorm",
            "--lognorm-target-sum", "7000",
            "--source-filter", "C05",
            "--target-filter", "C00",
            "--ligand-filter", "ADIPOQ",
            "--receptor-filter", "ADIPOR2",
            "--route-family-filter", "Adiponectin",
            "--max-edges", "25",
            "--min-sender-cells", "4",
            "--min-receiver-cells", "6",
            "--min-scored-donors-per-group", "2",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.ccc_backend == "liana_paired_rescore"
    assert cfg.ccc_condition_keys == ("sex@MASLD",)
    assert cfg.ccc_condition_values == ("yes",)
    assert cfg.ccc_compare_levels == ("female", "male")
    assert cfg.ccc_dataset_key == "tissue"
    assert cfg.ccc_source_levels == ("vfat",)
    assert cfg.ccc_target_levels == ("liv",)
    assert cfg.liana_candidate_events == "liana_rank_aggregate.tsv"
    assert cfg.liana_pairing_key == "sample_id"
    assert cfg.liana_input_mode == "lognorm"
    assert cfg.liana_lognorm_target_sum == pytest.approx(7000.0)
    assert cfg.liana_source_filter == ("C05",)
    assert cfg.liana_target_filter == ("C00",)
    assert cfg.liana_ligand_filter == ("ADIPOQ",)
    assert cfg.liana_receptor_filter == ("ADIPOR2",)
    assert cfg.liana_route_family_filter == ("Adiponectin",)
    assert cfg.liana_max_edges == 25
    assert cfg.liana_min_sender_cells == 4
    assert cfg.liana_min_receiver_cells == 6
    assert cfg.liana_min_scored_donors_per_group == 2


@patch("scomnom.cli.run_nichenet_ccc")
def test_ccc_nichenet_config_propagates(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "nichenet",
            "--input-path", "adata.zarr.tar.zst",
            "--condition-key", "sex@MASLD",
            "--condition-value", "yes",
            "--compare-level", "female",
            "--compare-level", "male",
            "--dataset-key", "tissue",
            "--source-level", "liver",
            "--target-level", "fat",
            "--signal-scope", "secreted",
            "--sender-cluster", "C00,C01",
            "--expression-pct", "0.2",
            "--input-mode", "lognorm",
            "--lognorm-target-sum", "7500",
            "--top-n-ligands", "12",
            "--top-n-targets", "80",
            "--min-logfc", "0.5",
            "--padj-threshold", "0.01",
            "--install-missing-r-deps",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.ccc_backend == "nichenet"
    assert cfg.ccc_condition_keys == ("sex@MASLD",)
    assert cfg.ccc_condition_values == ("yes",)
    assert cfg.ccc_compare_levels == ("female", "male")
    assert cfg.ccc_dataset_key == "tissue"
    assert cfg.ccc_source_levels == ("liver",)
    assert cfg.ccc_target_levels == ("fat",)
    assert cfg.ccc_signal_scope == "secreted"
    assert cfg.nichenet_receiver_cluster == "all"
    assert cfg.nichenet_sender_clusters == ("C00", "C01")
    assert cfg.nichenet_expression_pct == pytest.approx(0.2)
    assert cfg.nichenet_input_mode == "lognorm"
    assert cfg.nichenet_lognorm_target_sum == pytest.approx(7500.0)
    assert cfg.nichenet_top_n_ligands == 12
    assert cfg.nichenet_top_n_targets == 80
    assert cfg.nichenet_min_logfc == pytest.approx(0.5)
    assert cfg.nichenet_padj_threshold == pytest.approx(0.01)
    assert cfg.nichenet_install_missing_r_deps is True


@patch("scomnom.cli.run_mebocost_ccc")
def test_ccc_mebocost_config_propagates(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "mebocost",
            "--input-path", "adata.zarr.tar.zst",
            "--condition-key", "masld_status@timepoint",
            "--condition-value", "5_years_post",
            "--compare-level", "better",
            "--compare-level", "worse",
            "--dataset-key", "tissue",
            "--source-level", "liver",
            "--target-level", "fat",
            "--organism", "mouse",
            "--input-mode", "lognorm",
            "--lognorm-target-sum", "7500",
            "--n-shuffle", "250",
            "--seed", "123",
            "--min-cell-number", "15",
            "--pval-cutoff", "0.01",
            "--plot-top-n", "18",
            "--install-missing-python-deps",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.ccc_backend == "mebocost"
    assert cfg.ccc_condition_keys == ("masld_status@timepoint",)
    assert cfg.ccc_condition_values == ("5_years_post",)
    assert cfg.ccc_compare_levels == ("better", "worse")
    assert cfg.ccc_dataset_key == "tissue"
    assert cfg.ccc_source_levels == ("liver",)
    assert cfg.ccc_target_levels == ("fat",)
    assert cfg.mebocost_organism == "mouse"
    assert cfg.mebocost_input_mode == "lognorm"
    assert cfg.mebocost_lognorm_target_sum == pytest.approx(7500.0)
    assert cfg.mebocost_n_shuffle == 250
    assert cfg.mebocost_seed == 123
    assert cfg.mebocost_min_cell_number == 15
    assert cfg.mebocost_pval_cutoff == pytest.approx(0.01)
    assert cfg.mebocost_plot_top_n == 18
    assert cfg.mebocost_install_missing_python_deps is True


@patch("scomnom.cli.run_mebocost_paired_rescore")
def test_ccc_mebocost_paired_config_propagates(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "ccc",
            "mebocost-paired",
            "--input-path", "adata.zarr.tar.zst",
            "--candidate-events", "mebocost_sig_res.tsv",
            "--condition-key", "sex@MASLD",
            "--condition-value", "yes",
            "--compare-level", "female",
            "--compare-level", "male",
            "--dataset-key", "tissue",
            "--source-level", "vfat",
            "--target-level", "liv",
            "--pairing-key", "sample_id",
            "--organism", "human",
            "--input-mode", "lognorm",
            "--lognorm-target-sum", "6000",
            "--source-filter", "C05,C03",
            "--target-filter", "C00,C01",
            "--metabolite-filter", "HMDB0001",
            "--sensor-filter", "CD36",
            "--superclass-filter", "Lipids and lipid-like molecules",
            "--class-filter", "Fatty acyls",
            "--subclass-filter", "Fatty acids",
            "--max-events", "50",
            "--score-method", "associated-gene-proxy",
            "--min-sender-cells", "12",
            "--min-receiver-cells", "14",
            "--min-scored-donors-per-group", "4",
            "--install-missing-python-deps",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.ccc_backend == "mebocost_paired_rescore"
    assert cfg.mebocost_candidate_events == "mebocost_sig_res.tsv"
    assert cfg.ccc_condition_keys == ("sex@MASLD",)
    assert cfg.ccc_condition_values == ("yes",)
    assert cfg.ccc_compare_levels == ("female", "male")
    assert cfg.ccc_dataset_key == "tissue"
    assert cfg.ccc_source_levels == ("vfat",)
    assert cfg.ccc_target_levels == ("liv",)
    assert cfg.mebocost_pairing_key == "sample_id"
    assert cfg.mebocost_input_mode == "lognorm"
    assert cfg.mebocost_lognorm_target_sum == pytest.approx(6000.0)
    assert cfg.mebocost_source_filter == ("C05", "C03")
    assert cfg.mebocost_target_filter == ("C00", "C01")
    assert cfg.mebocost_metabolite_filter == ("HMDB0001",)
    assert cfg.mebocost_sensor_filter == ("CD36",)
    assert cfg.mebocost_superclass_filter == ("Lipids and lipid-like molecules",)
    assert cfg.mebocost_class_filter == ("Fatty acyls",)
    assert cfg.mebocost_subclass_filter == ("Fatty acids",)
    assert cfg.mebocost_max_events == 50
    assert cfg.mebocost_score_method == "associated-gene-proxy"
    assert cfg.mebocost_min_sender_cells == 12
    assert cfg.mebocost_min_receiver_cells == 14
    assert cfg.mebocost_min_scored_donors_per_group == 4
    assert cfg.mebocost_install_missing_python_deps is True


@patch("scomnom.cli.run_enrichment_cluster")
def test_enrichment_cluster_gene_filter_propagates_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "cluster",
            "--input-path", "adata.zarr.tar.zst",
            "--gene-filter", "not gene.str.startswith('MT-')",
            "--gene-filter", "expr=not gene.str.startswith('RPL')",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.gene_filter == (
        "not gene.str.startswith('MT-')",
        "expr=not gene.str.startswith('RPL')",
    )


@patch("scomnom.cli.run_enrichment_cluster")
def test_enrichment_cluster_condition_key_propagates_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "cluster",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r5_archetypes",
            "--condition-key", "sex",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.condition_key == "sex"


@patch("scomnom.cli.run_enrichment_de_from_tables")
def test_enrichment_de_default_output_name_uses_input_dir_name(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "de",
            "--input-dir", "de_r5_archetypes_round1",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "enrichment_de_de_r5_archetypes_round1"
    assert cfg.input_dir.name == "de_r5_archetypes_round1"


@patch("scomnom.cli.run_enrichment_de_from_tables")
def test_enrichment_de_gene_filter_propagates_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "de",
            "--input-dir", "de_r5_archetypes_round1",
            "--gene-filter", "not gene.str.startswith('MT-')",
            "--de-decoupler-source", "cell",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.gene_filter == ("not gene.str.startswith('MT-')",)
    assert cfg.de_decoupler_source == "cell"


@patch("scomnom.cli.run_module_score")
def test_enrichment_module_score_default_output_name_includes_round_and_set_name(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "module-score",
            "--input-path", "adata.zarr.tar.zst",
            "--round-id", "r5_archetypes",
            "--module-file", "archetypes.gmt",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.output_name == "adata.module_score_archetypes_r5_archetypes"
    assert cfg.module_set_name == "archetypes"
    assert cfg.module_files == ("archetypes.gmt",)


@patch("scomnom.cli.run_module_score")
def test_enrichment_module_score_method_propagates_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "enrichment",
            "module-score",
            "--input-path", "adata.zarr.tar.zst",
            "--module-file", "custom.tsv",
            "--module-set-name", "my_modules",
            "--module-score-method", "aucell",
            "--condition-key", "sex:MASLD",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.module_set_name == "my_modules"
    assert cfg.module_score_method == "aucell"
    assert cfg.condition_key == "sex:MASLD"


@patch("scomnom.cli.run_within_cluster")
def test_de_gene_filter_propagates_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "de",
            "--input-path", "adata.zarr.tar.zst",
            "--condition-keys", "sex",
            "--gene-filter", "not gene.str.startswith('MT-')",
            "--gene-filter", "expr=gene_biotype == 'protein_coding'",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.gene_filter == (
        "not gene.str.startswith('MT-')",
        "expr=gene_biotype == 'protein_coding'",
    )


@patch("scomnom.cli.run_within_cluster")
def test_de_gsea_flags_propagate_to_config(mock_run):
    result = runner.invoke(
        app,
        [
            "markers-and-de",
            "de",
            "--input-path", "adata.zarr.tar.zst",
            "--condition-keys", "sex",
            "--no-run-gsea",
            "--gsea-min-size", "15",
            "--gsea-max-size", "250",
            "--gsea-eps", "1e-6",
            "--gsea-rank-col", "log2FoldChange",
            "--joint-enrichment-alpha", "0.1",
            "--joint-enrichment-top-n", "12",
            "--no-joint-enrichment-require-concordant",
            "--no-joint-enrichment-require-gsea-sig",
            "--joint-enrichment-leading-edge-top-n", "5",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.run_gsea is False
    assert cfg.gsea_min_size == 15
    assert cfg.gsea_max_size == 250
    assert cfg.gsea_eps == pytest.approx(1e-6)
    assert cfg.gsea_rank_col == "log2FoldChange"
    assert cfg.joint_enrichment_alpha == pytest.approx(0.1)
    assert cfg.joint_enrichment_top_n == 12
    assert cfg.joint_enrichment_require_concordant is False
    assert cfg.joint_enrichment_require_gsea_sig is False
    assert cfg.joint_enrichment_leading_edge_top_n == 5
