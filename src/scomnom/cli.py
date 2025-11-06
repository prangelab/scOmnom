import typer
from .load_and_qc import run_load_and_qc
from .integrate import run_integration
from .cluster_and_annotate import run_clustering
from .config import LoadAndQCConfig

app = typer.Typer()

@app.command()
def load_and_qc(
    input_dir: str,
    output_dir: str,
    metadata_file: str = None,
    cellbender: bool = False,
):
    cfg = LoadAndQCConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_file=metadata_file,
        cellbender=cellbender,
    )
    run_load_and_qc(cfg)

@app.command()
def integrate():
    run_integration()

@app.command()
def cluster_and_annotate():
    run_clustering()

if __name__ == "__main__":
    app()