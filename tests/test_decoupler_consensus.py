from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scomnom.annotation_utils as au


def _estimate_frames() -> dict[str, pd.DataFrame]:
    sources = [f"S{i}" for i in range(6)]
    samples = ["C00", "C01", "C02"]
    first = pd.DataFrame(
        [
            [3.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
            [1.0, 4.0, 2.0],
            [-1.0, -3.0, -2.0],
            [-2.0, -1.0, -4.0],
            [-4.0, -2.0, -1.0],
        ],
        index=sources,
        columns=samples,
    )
    second = pd.DataFrame(
        [
            [1.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
            [2.0, 1.0, 3.0],
            [-3.0, -1.0, -2.0],
            [-1.0, -4.0, -3.0],
            [-2.0, -3.0, -1.0],
        ],
        index=sources,
        columns=samples,
    )
    return {"ulm": first, "mlm": second}


def test_decoupler_official_consensus_is_invariant_to_constituent_scale():
    import decoupler as dc

    estimates = _estimate_frames()
    baseline = au._dc_consensus_from_estimates(dc, estimates, verbose=False)
    scaled = au._dc_consensus_from_estimates(
        dc,
        {"ulm": estimates["ulm"], "mlm": estimates["mlm"] * 1000.0},
        verbose=False,
    )

    pd.testing.assert_frame_equal(baseline, scaled, rtol=1e-12, atol=1e-12)


def test_dc_run_method_maps_legacy_wsum_and_records_consensus_provenance(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def method(name, scale):
        def run(*, data, net, verbose, **kwargs):
            calls.append((name, kwargs))
            sources = sorted(net["source"].unique())
            values = np.arange(1, data.shape[0] * len(sources) + 1, dtype=float).reshape(
                data.shape[0], len(sources)
            )
            return pd.DataFrame(values * scale, index=data.index, columns=sources), None

        return run

    consensus_called = {}

    def consensus(result, verbose=False):
        consensus_called["keys"] = sorted(result)
        template = next(iter(result.values()))
        return pd.DataFrame(7.0, index=template.index, columns=template.columns), None

    fake = SimpleNamespace(
        mt=SimpleNamespace(
            ulm=method("ulm", 1.0),
            mlm=method("mlm", 100.0),
            waggr=method("waggr", 1000.0),
            consensus=consensus,
        )
    )
    monkeypatch.setitem(sys.modules, "decoupler", fake)

    mat = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]],
        index=["C00", "C01"],
        columns=["G1", "G2", "G3"],
    )
    net = pd.DataFrame(
        {
            "source": ["P1", "P1", "P2", "P2"],
            "target": ["G1", "G2", "G2", "G3"],
            "weight": [1.0, 1.0, -1.0, 1.0],
        }
    )

    result = au._dc_run_method(
        method="consensus",
        mat=mat,
        net=net,
        min_n=1,
        consensus_methods=["ulm", "mlm", "wsum"],
    )

    assert np.all(result.to_numpy() == 7.0)
    assert consensus_called["keys"] == ["score_mlm", "score_ulm", "score_wsum"]
    assert ("waggr", {"tmin": 1, "fun": "wsum", "times": 0}) in calls
    provenance = result.attrs["method_provenance"]
    assert provenance["combiner"] == "decoupler.mt.consensus_signed_zscore"
    assert provenance["successful_constituents"] == ["ulm", "mlm", "wsum"]
    assert provenance["resolved_constituents"]["wsum"]["resolved"] == "waggr"


def test_dc_run_method_refuses_single_method_consensus(monkeypatch):
    def ulm(*, data, net, verbose, **kwargs):
        sources = sorted(net["source"].unique())
        return pd.DataFrame(1.0, index=data.index, columns=sources), None

    fake = SimpleNamespace(mt=SimpleNamespace(ulm=ulm, consensus=lambda result, verbose=False: result))
    monkeypatch.setitem(sys.modules, "decoupler", fake)
    mat = pd.DataFrame([[1.0, 2.0]], index=["C00"], columns=["G1", "G2"])
    net = pd.DataFrame(
        {"source": ["P1", "P1"], "target": ["G1", "G2"], "weight": [1.0, 1.0]}
    )

    with pytest.raises(RuntimeError, match="fewer than two"):
        au._dc_run_method(
            method="consensus",
            mat=mat,
            net=net,
            min_n=1,
            consensus_methods=["ulm", "missing"],
        )


def test_dc_run_method_smoke_with_installed_decoupler_consensus():
    rng = np.random.default_rng(42)
    genes = [f"G{i}" for i in range(8)]
    mat = pd.DataFrame(rng.normal(size=(5, len(genes))), index=[f"C{i}" for i in range(5)], columns=genes)
    net = pd.DataFrame(
        {
            "source": ["P1"] * 4 + ["P2"] * 4,
            "target": genes,
            "weight": [1.0, 0.5, -0.5, 1.0, -1.0, 0.5, 1.0, -0.5],
        }
    )

    result = au._dc_run_method(
        method="consensus",
        mat=mat,
        net=net,
        min_n=2,
        consensus_methods=["ulm", "mlm", "wsum"],
    )

    assert result.shape == (2, 5)
    assert np.isfinite(result.to_numpy()).all()
    provenance = result.attrs["method_provenance"]
    assert provenance["successful_constituents"] == ["ulm", "mlm", "wsum"]
    assert provenance["resolved_constituents"]["wsum"]["kwargs"] == {
        "tmin": 2,
        "fun": "wsum",
        "times": 0,
    }
