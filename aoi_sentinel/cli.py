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
@click.option("--labels", required=True, type=click.Path(exists=True), help="Parquet/CSV stream of EvalRecord rows")
@click.option("--out", default="eval_report.json", show_default=True, help="Where to write the report JSON")
@click.option("--cost-escape", default=1000.0, show_default=True, type=float)
@click.option("--cost-false-call", default=1.0, show_default=True, type=float)
@click.option("--cost-operator", default=5.0, show_default=True, type=float)
def eval(labels: str, out: str, cost_escape: float, cost_false_call: float, cost_operator: float) -> None:
    """Run the eval pipeline on a labelled stream and emit the report.

    Karpathy first-principle: eval before train. This command runs even
    before any model is trained — point it at the vendor-only baseline
    output to get the ceiling cost number on day zero.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime

    from aoi_sentinel.eval.runner import EvalRecord, run_eval

    p = pd.read_parquet(labels) if labels.endswith(".parquet") else pd.read_csv(labels)
    required = {"timestamp", "vendor_call", "engine_action", "engine_confidence", "label"}
    missing = required - set(p.columns)
    if missing:
        raise click.UsageError(f"input missing columns: {missing}")

    records = [
        EvalRecord(
            timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else datetime.fromisoformat(str(row["timestamp"])),
            vendor_call=str(row["vendor_call"]),
            engine_action=str(row["engine_action"]),
            engine_confidence=float(row["engine_confidence"]),
            label=str(row["label"]),
        )
        for _, row in p.iterrows()
    ]

    cm = np.zeros((2, 3), dtype=np.float32)
    cm[0, 0] = cost_false_call; cm[0, 1] = 0.0;          cm[0, 2] = cost_operator
    cm[1, 0] = 0.0;             cm[1, 1] = cost_escape;  cm[1, 2] = cost_operator

    report = run_eval(records, cost_matrix=cm)
    report.to_json(out)

    eng = report.engine; bl = report.baseline_vendor_only; d = report.delta
    console.print(f"[bold]eval done[/bold]  n_total={report.n_total}  n_labeled={report.n_labeled}")
    console.print(f"  baseline   cost={bl.get('expected_cost', 0):.3f}  fc={bl.get('false_call_rate', 0)*100:.2f}%  escape={bl.get('escape_rate', 0)*100:.4f}%")
    console.print(f"  engine     cost={eng.get('expected_cost', 0):.3f}  fc={eng.get('false_call_rate', 0)*100:.2f}%  escape={eng.get('escape_rate', 0)*100:.4f}%  AURC={report.aurc_engine:.4f}")
    console.print(f"  [green]Δ cost[/green] = {d.get('expected_cost_drop', 0):+.3f}   [green]Δ fc[/green] = {d.get('false_call_drop_pp', 0)*100:+.2f}pp")
    console.print(f"  → report: {out}")


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Run the FastAPI inference server."""
    import uvicorn

    uvicorn.run("aoi_sentinel.serve.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
