from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List

class LoadAndQCConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    metadata_file: Optional[Path] = None
    cellbender: bool = False
    min_genes: int = 200
    min_cells: int = 3
    mt_gene_prefix: str = "MT-"
    ribo_gene_prefix: str = "RPS"
    hb_gene_prefix: str = "HB"
    n_top_genes: int = 2000
    n_pcs: int = 50
    figures: bool = True