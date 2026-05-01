"""aoi-sentinel CLI entrypoint."""
from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """aoi-sentinel — Mamba-RL false-call filter for SMT AOI."""


@main.group()
def data() -> None:
    """Data ingestion and preparation."""


@data.command("scan")
@click.option("--root", required=True, type=click.Path(exists=True), help="Saki dump root")
@click.option("--out", default="data/index.parquet", show_default=True)
def data_scan(root: str, out: str) -> None:
    """Scan a real Saki inspection dump and build an index parquet."""
    from aoi_sentinel.data.saki import scan_saki_dump

    n = scan_saki_dump(root, out)
    console.print(f"[green]indexed {n} samples → {out}[/green]")


@main.group()
def train() -> None:
    """Training entrypoints."""


@train.command("pretrain")
@click.option("--config", required=True, type=click.Path(exists=True))
def train_pretrain(config: str) -> None:
    """Stage 0 — supervised pretraining of the MambaVision image encoder."""
    from aoi_sentinel.train.stage0_pretrain import run

    run(config)


@train.command("npi-rl")
@click.option("--config", required=True, type=click.Path(exists=True))
def train_npi_rl(config: str) -> None:
    """Stage 1 — Lagrangian PPO on the NPI simulator."""
    from aoi_sentinel.train.stage1_npi_rl import run

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
