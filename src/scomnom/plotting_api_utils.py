from __future__ import annotations

from functools import wraps
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable

from . import de_plot_utils, plot_utils

_HIDDEN_INTERNAL_PARAMS = {
    "figdir",
    "cfg",
    "stage",
    "stem",
    "artifact_stem",
    "artifact_figdir",
    "savefig_kwargs",
    "show",
    "plot_name",
    "figure_formats",
    "fname",
    "legend_figdir",
    "legend_stem",
    "suffix",
}


def _collect_artifacts(result, captured: Iterable[plot_utils.PlotArtifact]) -> list[plot_utils.PlotArtifact]:
    out: list[plot_utils.PlotArtifact] = []
    if isinstance(result, plot_utils.PlotArtifact):
        out.append(result)
    elif isinstance(result, list) and all(isinstance(x, plot_utils.PlotArtifact) for x in result):
        out.extend(result)
    out.extend(list(captured))

    seen: set[int] = set()
    uniq: list[plot_utils.PlotArtifact] = []
    for art in out:
        key = id(art)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(art)
    return uniq


def _build_public_signature(plot_fn: Callable) -> inspect.Signature:
    sig = inspect.signature(plot_fn)
    kept: list[inspect.Parameter] = []
    var_kw: inspect.Parameter | None = None
    for p in sig.parameters.values():
        if p.name in _HIDDEN_INTERNAL_PARAMS:
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw = p
            continue
        kept.append(p)

    kept.extend(
        [
            inspect.Parameter(
                "display",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=True,
                annotation=bool,
            ),
            inspect.Parameter(
                "file",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=str | Path | None,
            ),
            inspect.Parameter(
                "return_fig",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            ),
        ]
    )
    if var_kw is not None:
        kept.append(var_kw)
    return sig.replace(parameters=kept)


def _inject_api_defaults(plot_fn: Callable, kwargs: dict) -> dict:
    """
    Fill internal-only parameters for API calls so users don't need CLI plumbing args.
    """
    try:
        sig = inspect.signature(plot_fn)
    except Exception:
        return kwargs

    out = dict(kwargs)
    blocked = sorted([k for k in out.keys() if k in _HIDDEN_INTERNAL_PARAMS])
    if blocked:
        raise TypeError(
            f"Internal parameter(s) not exposed in API wrapper: {', '.join(blocked)}."
        )

    if "figdir" in sig.parameters and "figdir" not in out:
        out["figdir"] = Path(".")
    if "cfg" in sig.parameters and "cfg" not in out:
        out["cfg"] = SimpleNamespace(
            make_figures=True,
            batch_key=out.get("batch_key", None),
        )
    if "stage" in sig.parameters and "stage" not in out:
        out["stage"] = "api"
    if "show" in sig.parameters and "show" not in out:
        out["show"] = False
    if "artifact_stem" in sig.parameters and "artifact_stem" not in out:
        out["artifact_stem"] = None
    if "artifact_figdir" in sig.parameters and "artifact_figdir" not in out:
        out["artifact_figdir"] = None
    if "savefig_kwargs" in sig.parameters and "savefig_kwargs" not in out:
        out["savefig_kwargs"] = None
    if "legend_figdir" in sig.parameters and "legend_figdir" not in out:
        out["legend_figdir"] = None
    if "legend_stem" in sig.parameters and "legend_stem" not in out:
        out["legend_stem"] = "legend"
    if "suffix" in sig.parameters and "suffix" not in out:
        out["suffix"] = "api"
    return out


def _run_plot_fn(plot_fn: Callable, *args, **kwargs) -> list[plot_utils.PlotArtifact]:
    kwargs = _inject_api_defaults(plot_fn, kwargs)
    with plot_utils.keep_plots_open():
        with plot_utils.capture_plot_artifacts() as captured:
            result = plot_fn(*args, **kwargs)
    return _collect_artifacts(result, captured)


def _collect_figures(artifacts: Iterable[plot_utils.PlotArtifact]) -> list:
    figs = []
    seen: set[int] = set()
    for art in artifacts:
        fig = getattr(art, "fig", None)
        if fig is None:
            continue
        fid = id(fig)
        if fid in seen:
            continue
        seen.add(fid)
        figs.append(fig)
    return figs


def _display_figures(figs: Iterable) -> None:
    try:
        from IPython.display import display as ipy_display
    except Exception:
        ipy_display = None

    for fig in figs:
        if ipy_display is not None:
            ipy_display(fig)
        else:
            try:
                fig.show()
            except Exception:
                pass


def _save_artifacts_to_file(
    artifacts: list[plot_utils.PlotArtifact],
    file: str | Path,
) -> None:
    out = Path(file)
    ext = out.suffix.lstrip(".")
    if not ext:
        raise ValueError("`file` must include an extension, for example: 'figure.png'.")
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(artifacts) == 1:
        art = artifacts[0]
        fig = getattr(art, "fig", None)
        if fig is None:
            return
        kwargs = dict(art.savefig_kwargs) if isinstance(art.savefig_kwargs, dict) else {}
        kwargs.setdefault("format", ext)
        fig.savefig(out, **kwargs)
        return

    base = out.stem
    for art in artifacts:
        fig = getattr(art, "fig", None)
        if fig is None:
            continue
        outfile = out.parent / f"{base}__{art.stem}.{ext}"
        kwargs = dict(art.savefig_kwargs) if isinstance(art.savefig_kwargs, dict) else {}
        kwargs.setdefault("format", ext)
        fig.savefig(outfile, **kwargs)


def _format_api_return(figs: list, return_fig: bool):
    if not bool(return_fig):
        return None
    if len(figs) == 1:
        return figs[0]
    return figs


def _make_api_wrapper(public_name: str, plot_fn: Callable):
    public_sig = _build_public_signature(plot_fn)

    @wraps(plot_fn)
    def _wrapped(*args, display: bool = True, file: str | Path | None = None, return_fig: bool = False, **kwargs):
        artifacts = _run_plot_fn(plot_fn, *args, **kwargs)
        figs = _collect_figures(artifacts)
        if file is not None and str(file).strip() != "":
            _save_artifacts_to_file(artifacts, file=file)
        if bool(display):
            _display_figures(figs)
            # In notebooks, open figures are auto-rendered at cell end.
            # Close displayed figures when caller does not request figure returns
            # to avoid duplicate output.
            if not bool(return_fig):
                for fig in figs:
                    try:
                        plot_utils.close_plot(fig)
                    except Exception:
                        pass
        return _format_api_return(figs, return_fig=bool(return_fig))

    _wrapped.__name__ = public_name
    _wrapped.__qualname__ = public_name
    _wrapped.__doc__ = (
        f"API wrapper for `{plot_fn.__module__}.{plot_fn.__name__}`. "
        "Supports notebook display and optional direct path saving."
    )
    _wrapped.__signature__ = public_sig
    return _wrapped


_WRAPPER_SPECS: list[tuple[str, Callable]] = [
    ("plot_de_heatmap_top_genes_by_sample", de_plot_utils.heatmap_top_genes_by_sample),
    ("plot_de_volcano", de_plot_utils.volcano),
    ("plot_de_dotplot_top_genes", de_plot_utils.dotplot_top_genes),
    ("plot_de_heatmap_top_genes", de_plot_utils.heatmap_top_genes),
    ("plot_de_violin_grid_genes", de_plot_utils.violin_grid_genes),
    ("plot_de_violin_genes", de_plot_utils.violin_genes),
    ("plot_de_umap_features_grid", de_plot_utils.umap_features_grid),
    ("plot_de_umap_single", de_plot_utils.umap_single),
]


for _public_name, _plot_fn in _WRAPPER_SPECS:
    globals()[_public_name] = _make_api_wrapper(_public_name, _plot_fn)


__all__ = [name for name, _ in _WRAPPER_SPECS]
