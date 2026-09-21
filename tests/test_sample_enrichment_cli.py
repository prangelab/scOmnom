from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest
from typer.main import get_command
from typer.testing import CliRunner

cli = importlib.import_module("scomnom.cli")
runner = CliRunner()


@pytest.fixture
def dispatch(monkeypatch):
    run = Mock()
    monkeypatch.setattr(cli, "run_sample_enrichment", run)
    return run


def test_sample_help_describes_replicates_and_explicit_pairing():
    result = runner.invoke(cli.app, ["enrichment", "sample", "--help"], terminal_width=160)
    assert result.exit_code == 0, result.output
    for option in ("--input-path", "--replicate-key", "--condition-key", "--subject-key", "--contrast", "--counts-layer"):
        assert option in result.output
    assert "sample_id" in result.output
    assert "donor_id" in result.output
    assert "figures" in result.output.lower()


def test_sample_is_only_registered_on_canonical_route():
    root = get_command(cli.app)
    canonical = root.commands["enrichment"]
    legacy = root.commands["markers-and-de"].commands["enrichment"]
    assert set(canonical.commands) == {"cluster", "de", "module-score", "sample"}
    assert set(legacy.commands) == {"cluster", "de", "module-score"}
    for name in legacy.commands:
        assert legacy.commands[name].callback.__wrapped__ is canonical.commands[name].callback.__wrapped__
        assert legacy.commands[name].help == canonical.commands[name].help
    assert runner.invoke(cli.app, ["markers-and-de", "enrichment", "sample", "--help"]).exit_code != 0


@pytest.mark.parametrize("args", [[], ["--input-path", "input.zarr"], ["--replicate-key", "sample"]])
def test_sample_requires_input_and_replicate_key(args, dispatch):
    result = runner.invoke(cli.app, ["enrichment", "sample", *args])
    assert result.exit_code == 2
    dispatch.assert_not_called()


def test_regeneration_uses_only_rendering_options(dispatch):
    result = runner.invoke(cli.app, ["enrichment", "sample", "-i", "saved.zarr", "--regenerate-figures",
                                     "--round-id", "r1", "--analysis-id", "enrichment_sample_r1_round2", "-F", "png"])
    assert result.exit_code == 0, result.output
    cfg = dispatch.call_args.args[0]
    assert cfg.regenerate_figures and cfg.replicate_key is None and cfg.output_name is None
    assert cfg.analysis_id == "enrichment_sample_r1_round2"


@pytest.mark.parametrize("extra", [["--covariates", "age"], ["--replicate-key", "sample"],
                                  ["--no-make-figures"], ["--save-h5ad"], ["--counts-layer", "auto"]])
def test_regeneration_rejects_analysis_overrides(extra, dispatch):
    result = runner.invoke(cli.app, ["enrichment", "sample", "-i", "saved.zarr", "--regenerate-figures", *extra])
    assert result.exit_code == 2, result.output
    dispatch.assert_not_called()


def test_independent_dispatch_records_exact_option_tokens(dispatch, tmp_path):
    path = tmp_path / "results" / "nested" / "input.zarr"
    args = [
        "enrichment", "sample", "-i", str(path), "--round-id", "r1",
        "--replicate-key", "sample", "--condition-key", "sex", "--contrast", "female:male",
        "--covariates", "age,BMI", "--covariates", "batch", "--target-groups", "C03,C04",
        "--plot-activity", "STAT2,IRF9", "--dorothea-method", "ulm", "--save-h5ad",
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    cfg = dispatch.call_args.args[0]
    assert cfg.output_dir == tmp_path / "results"
    assert cfg.output_name == "adata.enrichment_sample_r1"
    assert cfg.contrasts == ("female:male",)
    assert cfg.covariates == ("age", "BMI", "batch")
    assert cfg.target_groups == ("C03", "C04")
    assert cfg.plot_activity == ("STAT2", "IRF9")
    assert cfg.save_h5ad is True
    assert cfg.dorothea_method == "ulm"
    assert dispatch.call_args.kwargs["command"] == ["scomnom", *args]


def test_paired_dispatch_and_all_thresholds(dispatch):
    result = runner.invoke(cli.app, [
        "enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample_id",
        "--condition-key", "condition", "--reference", "ctrl", "--covariates", "donor_id",
        "--subject-key", "donor_id", "--min-cells-per-replicate-group", "30",
        "--min-replicates-per-level", "4", "--min-replicates-total", "8", "--min-complete-subjects", "4",
    ])
    assert result.exit_code == 0, result.output
    cfg = dispatch.call_args.args[0]
    assert cfg.subject_key == "donor_id"
    assert cfg.covariates == ("donor_id",)
    assert cfg.reference == "ctrl"
    assert cfg.min_cells_per_replicate_group == 30
    assert cfg.min_replicates_per_level == cfg.min_complete_subjects == 4
    assert cfg.min_replicates_total == 8


def test_resource_controls_and_gene_filters_dispatch_without_csv_splitting_queries(dispatch):
    result = runner.invoke(cli.app, [
        "enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample",
        "--no-run-msigdb", "--no-run-progeny", "--decoupler-method", "consensus",
        "--decoupler-consensus-methods", "ulm,mlm", "--dorothea-confidence", "A,B",
        "--dorothea-organism", "mouse", "--dorothea-min-n-targets", "8", "--counts-layer", "counts_raw",
        "--gene-filter", "gene in ['A', 'B']", "--figure-formats", "png,pdf",
    ])
    assert result.exit_code == 0, result.output
    cfg = dispatch.call_args.args[0]
    assert not cfg.run_msigdb and not cfg.run_progeny
    assert cfg.decoupler_consensus_methods == ["ulm", "mlm"]
    assert cfg.dorothea_confidence == ["A", "B"]
    assert cfg.dorothea_organism == "mouse"
    assert cfg.dorothea_min_n_targets == 8
    assert cfg.counts_layer == "counts_raw"
    assert cfg.gene_filter == ("gene in ['A', 'B']",)
    assert cfg.figure_formats == ["png", "pdf"]


@pytest.mark.parametrize("extra", [
    ["--condition-key", "condition", "--contrast", "bad"],
    ["--condition-key", "condition", "--contrast", "ctrl:ctrl"],
    ["--contrast", "stim:ctrl"], ["--subject-key", "donor"],
    ["--min-replicates-per-level", "0"], ["--counts-layer", "raw"],
    ["--decoupler-consensus-methods", "ulm"], ["--dorothea-confidence", "Z"],
])
def test_invalid_options_fail_before_dispatch(extra, dispatch):
    result = runner.invoke(cli.app, ["enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample", *extra])
    assert result.exit_code == 2, result.output
    dispatch.assert_not_called()


def test_default_active_round_name_is_resolved_after_loading(dispatch):
    result = runner.invoke(cli.app, ["enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample"])
    assert result.exit_code == 0, result.output
    cfg = dispatch.call_args.args[0]
    assert cfg.round_id is None and cfg.output_name is None
    assert str(cfg.output_dir) == "results"
    assert cfg.counts_layer == "auto"
    assert cfg.msigdb_gene_sets == ["HALLMARK", "REACTOME"]
    assert cfg.dorothea_method is None


def test_multiple_contrasts_are_repeatable_and_csv_compatible(dispatch):
    result = runner.invoke(cli.app, [
        "enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample",
        "--condition-key", "condition", "--contrast", "a:b,c:b", "--contrast", "d:b",
    ])
    assert result.exit_code == 0, result.output
    assert dispatch.call_args.args[0].contrasts == ("a:b", "c:b", "d:b")


@pytest.mark.parametrize("error,code", [(ValueError("population C99 is unavailable"), 2), (RuntimeError("resource failed"), 1)])
def test_analysis_failures_are_reported(dispatch, error, code):
    dispatch.side_effect = error
    result = runner.invoke(cli.app, ["enrichment", "sample", "-i", "input.zarr", "--replicate-key", "sample"])
    assert result.exit_code == code
    assert str(error) in result.output
