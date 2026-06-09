"""
app/engine/oed -- Optimal Experimental Design for Physiological Parameter Identification.

FASE 3 -- Active Inference via Fisher Information.

Exports
-------
Pilar 1 -- FIM Evaluator (identifiability.py)
    FIMResult              : identifiability report for one (theta, protocol) pair
    SliceFIMConfig         : configuration bundle for a slice's FIM evaluation
    compute_fim            : core FIM = sum_t J_t.T @ R_inv @ J_t via jax.jacfwd
    analyze_fim            : eigendecompose FIM, label unidentifiable directions
    evaluate_identifiability: one-call convenience wrapper

Pilar 3 -- OED Protocol Generator (fisher_design.py)
    OEDAction              : result of protocol selection (winner + all FIM results)
    OEDProtocolGenerator   : rank candidate protocols by D-optimality log det(FIM)
    make_cardio_oed_generator       : pre-built generator for Cardiorespiratory slice
    make_neuromuscular_oed_generator: pre-built generator for Neuromuscular slice
    make_cardio_fim_factory         : forward factory (x0, power) -> (log_theta -> HR)
    make_neuromuscular_fim_factory  : forward factory (x0, ctrl) -> (log_theta -> y)
"""
from app.engine.oed.identifiability import (
    FIMResult,
    SliceFIMConfig,
    compute_fim,
    analyze_fim,
    evaluate_identifiability,
)
from app.engine.oed.fisher_design import (
    OEDAction,
    OEDProtocolGenerator,
    make_cardio_oed_generator,
    make_neuromuscular_oed_generator,
    make_cardio_fim_factory,
    make_neuromuscular_fim_factory,
)

__all__ = [
    "FIMResult",
    "SliceFIMConfig",
    "compute_fim",
    "analyze_fim",
    "evaluate_identifiability",
    "OEDAction",
    "OEDProtocolGenerator",
    "make_cardio_oed_generator",
    "make_neuromuscular_oed_generator",
    "make_cardio_fim_factory",
    "make_neuromuscular_fim_factory",
]
