"""aoi-sentinel CLI entrypoint."""
from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """aoi-sentinel — SMT AOI false-call filter and defect analyzer."""


@main.group()
def data() -> None:
    """Data ingestion and preparation."""


@data.command("scan")
@click.option("--root", required=True, type=click.Path(exists=True), help="Saki dump root")
@click.option("--out", default="data/index.parquet", show_default=True)
def data_scan(root: str, out: str) -> None:
    """Scan Saki inspection dump and build an index parquet."""
    from aoi_sentinel.data.saki import scan_saki_dump

    n = scan_saki_dump(root, out)
    console.print(f"[green]indexed {n} samples → {out}[/green]")


@main.group()
def train() -> None:
    """Training entrypoints."""


@train.command("classifier")
@click.option("--config", required=True, type=click.Path(exists=True))
def train_classifier(config: str) -> None:
    """Train the 2D false-call classifier."""
    from aoi_sentinel.train.classifier_2d import run

    run(config)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Run the FastAPI inference server."""
    import uvicorn

    uvicorn.run("aoi_sentinel.serve.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
