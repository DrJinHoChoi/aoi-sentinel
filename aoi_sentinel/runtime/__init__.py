"""Runtime — the always-on parts of the system.

Two processes, both on-prem:

    edge.py             — runs on the line edge box (Jetson Orin Nano).
                          Adapter watch → infer → operator UI → label queue.

    trainer_server.py   — runs on the on-prem trainer box (DGX Spark).
                          Pulls labels from edges → continual learning →
                          safety gate → atomic model promotion.

Coordination glue:

    label_queue   — append-only label store, sync'd between edge ↔ trainer.
    model_registry— versioned model cache + atomic swap.
    safety_gate   — escape-rate hold-out check before any promotion.
    modes         — SHADOW / ASSIST / AUTONOMOUS state machine.
"""
