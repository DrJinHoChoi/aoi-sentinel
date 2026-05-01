"""NPI simulator — replays benchmark data as a streaming Saki line for safe RL training.

Components:
    NpiEnv         — gymnasium env with {DEFECT, PASS, ESCALATE} actions and a hard escape constraint
    LabelOracle    — simulated operator that reveals ground truth on ESCALATE (and lazily otherwise)
    CostMatrix     — pluggable cost matrix; default mirrors automotive escape costs
"""
from aoi_sentinel.sim.cost import CostMatrix, default_cost_matrix
from aoi_sentinel.sim.label_oracle import LabelOracle
from aoi_sentinel.sim.npi_env import NpiEnv

__all__ = ["NpiEnv", "LabelOracle", "CostMatrix", "default_cost_matrix"]
