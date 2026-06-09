# AUDIT_REPORT.md — Nutrivious Aerobic/HRV Vertical Slice

**Date:** 2026-06-09  
**Auditor:** Claude Code (read-only pass)  
**Branch:** main  
**Scope:** Full codebase scan with depth focus on the cardiorespiratory slice, engine layer, control layer, and validation layer.  
**Authority:** This is a READ-ONLY report. No code was modified. No files were deleted. Awaiting explicit OK before any write action.

---

## EXECUTIVE SUMMARY

The aerobic/HRV vertical slice has a **well-designed mechanical core** (6-state ODE, UKF, NLME, envelope) that is **architecturally isolated and cannot run end-to-end**. The primary blockers are:

1. **Two incompatible state architectures** co-exist with no bridge: the old 9-state observer (referenced by safety filter, NMPC, lexical filter, inference layer) and the new 6-state cardiorespiratory slice (referenced only by the three existing tests).
2. **Four modules fail to import at load time** due to broken symbol references to `app.engine.assimilation.ukf_filter` (which only exports `GaussianState`) and to `app/engine/solvers/` and `app/validation/` directories that do not exist.
3. **Gate Zero does not exist as code.** The most critical production-promotion criterion has zero implementation.
4. **One mathematical incoherence** in the UKF filter that violates Gaussian coherence post-clamping.

The 3 existing tests (T1–T3) pass physics sanity checks on the new 6-state slice but do not constitute predictive validation. No walk-forward harness exists.

---

## SECTION 1 — INVENTORY: REAL vs FAKE

### 1.1 REAL (code delivers what docstring/header promises)

| Module | What it delivers |
|--------|-----------------|
| `app/slices/cardiorespiratory/ode.py` | 6-state ODE fully implemented. JIT+vmap safe. NaN guards on hub inputs. Literature-cited parameters. Clean. |
| `app/slices/cardiorespiratory/filter.py` | UKF predict+update kernels implemented in pure JAX. α=0.10 justified. Fail-loud on NaN posterior. **One bug** (see B-02). |
| `app/slices/cardiorespiratory/nlme.py` | NLME with non-centred parametrisation and NumPyro SVI implemented. Fail-loud on ELBO NaN. |
| `app/slices/cardiorespiratory/observation.py` | Observation model `h_cardio`, NaN inflation, quality-flag R inflation — all implemented. |
| `app/slices/cardiorespiratory/envelope.py` | `Phase3Envelope` construction: hard constraints, chance constraints, refer-out rules implemented. **One silent-failure risk** (see B-05). |
| `app/engine/priors.py` | `build_engine_priors` fully implemented. SNP lookups, HR_max age-scaling, all correct. Zero imports. |
| `app/engine/phase3_envelope.py` | NamedTuples and `build_phase3_envelope` implemented. Safety ceilings match HARD_CEILINGS. |
| `app/engine/unit_converter.py` | SI normalisation for 3 analytes implemented with correct NIST conversion factors. |
| `app/engine/assimilation/ukf_filter.py` | Correctly provides `GaussianState`. That is all it promises (all other symbols were promised by callers, not by this module). |
| `app/engine/observation/aerobic_observer.py` | 9-state observer ODE + observation model implemented (old architecture — may be archived). |
| `app/compliance/lexical_filter.py` | `PathAFilter` regex gate and `assert_path_a` implemented. **One broken import** (see B-04). |
| `tests/test_cardiorespiratory_slice.py` | T1–T3 implemented and cover ODE physics + UKF stability. Pass on new architecture. |

### 1.2 FAKE (docstring/header promises, code does not deliver)

| Module | What it claims | What it actually does |
|--------|---------------|----------------------|
| `app/slices/aerobic_pipeline.py` | "Wires L2→L8 aerobic pipeline, AerobicSliceOrchestrator" | Fails at import time. References 4 non-existent modules. Marked DEPRECATED. |
| `app/control/safety_filter.py` | "Predictive Safety Filter (Wabersich-Zeilinger)" | Fails at import time. Also uses wrong state indices if import were fixed. |
| `app/inference/nlme_model.py` | "AerobicNLME personalises 9-state observer" | Imports `X0_DEFAULT` from `ukf_filter` (doesn't exist). Also silently returns wrong personalisation values (wrong key in `get_individual_theta`). |
| `app/validation/` | (required by `aerobic_pipeline.py`: `backtest_engine.py`) | **Directory does not exist.** Gate Zero has no implementation. |
| `app/engine/solvers/` | (required by `aerobic_pipeline.py`: `cardiorespiratory_solver.py`) | **Directory does not exist.** |
| End-to-end orchestrator | No file exists that wires cardio slice L2→L8 | No canonical orchestrator for the 6-state architecture. `aerobic_pipeline.py` is DEPRECATED and broken. |

---

## SECTION 2 — BUGS RANKED BY SEVERITY

### TIER 0 — CRITICAL SAFETY (violates I4: safety as hard constraint)

None found at this tier in the new 6-state architecture. The ODE hard constraints (HR ceiling, W'_bal floor, RF ceiling) are correctly wired.

**Note on safety_filter.py:** The PSF *architecture* is sound (Wabersich-Zeilinger design), but the module cannot import (see B-01), so it provides zero safety coverage in practice.

---

### TIER 1 — RUNTIME BREAK (ImportError at load time)

#### B-01 — `app/control/safety_filter.py`: broken import, wrong state indices
**Severity:** CRITICAL — runtime break + silent math error if fixed naively  
**Location:** `safety_filter.py`, top-level imports  
**Detail:**
```python
from app.engine.assimilation.ukf_filter import (
    _ukf_predict,          # does not exist — only GaussianState is exported
    Q_DEFAULT,             # does not exist
    AerobicTransitionParams # does not exist
)
from app.engine.observation.aerobic_observer import IDX_V_VAGAL, IDX_W_PRIME
# IDX_V_VAGAL = 2, IDX_W_PRIME = 8  ← 9-state indices
# But CardioStateFilter uses 6-state: AT=5, W_prime=3
```
This module fails at Python import. Even if the missing symbols were provided, the state indices are wrong for the 6-state architecture.

**Fix required (await OK):** (a) Import `GaussianState` and `CardioTransitionParams` from the correct modules; (b) write a `_cardio_ukf_predict` wrapper using `CardioStateFilter._ukf_predict_cardio`; (c) update indices to `IDX_AT=5`, `IDX_WPRIME=3`.

---

#### B-02-a — `app/slices/aerobic_pipeline.py`: broken imports × 4
**Severity:** CRITICAL — runtime break (DEPRECATED file, but still importable by other modules)  
**Location:** `aerobic_pipeline.py`, top-level imports  
**Detail:**
- `from app.engine.solvers.cardiorespiratory_solver import CardiorespiratorySolver` — `app/engine/solvers/` does not exist
- `from app.validation.backtest_engine import ValidationHarness, GateZeroDecision` — `app/validation/` does not exist
- `from app.engine.assimilation.ukf_filter import AerobicStateFilter, AerobicTransitionParams, X0_DEFAULT, P0_DEFAULT` — none of these exist in `ukf_filter.py`
- Uses 9-state imports from `aerobic_observer.py` (`IDX_V_VAGAL`, `IDX_W_PRIME`)

This file must be archived (not deleted — proposal, await OK) and replaced by a 6-state canonical orchestrator.

---

#### B-03 — `app/inference/nlme_model.py`: broken import
**Severity:** CRITICAL — runtime break  
**Location:** `nlme_model.py` line ~555  
**Detail:**
```python
from app.engine.assimilation.ukf_filter import X0_DEFAULT
```
`X0_DEFAULT` does not exist in `ukf_filter.py`. Module fails to load.

**Additional bug (silent, same file):** `get_individual_theta()` calls `posterior_samples.get("scale_eta_log", ...)` but NumPyro samples `scale_eta` via HalfNormal (not in log space). Returns None silently for the off-diagonal scale parameters → wrong individual personalisation with no exception.

---

#### B-04 — `app/compliance/lexical_filter.py`: broken import
**Severity:** CRITICAL — runtime break  
**Location:** `lexical_filter.py`, imports  
**Detail:**
```python
from app.engine.assimilation.ukf_filter import AerobicTransitionParams
```
`AerobicTransitionParams` does not exist in `ukf_filter.py`. The compliance module — which must run on every output — fails at import.

---

### TIER 2 — MATH INCONSISTENCY (violates I2: probabilistic; produces wrong state)

#### B-05 — `_apply_physical_clamps` breaks Gaussian coherence
**Severity:** HIGH — mathematical incoherence; violates invariant I2 and I5  
**Location:** `app/slices/cardiorespiratory/filter.py`, `_apply_physical_clamps()`  
**Detail:**
```python
def _apply_physical_clamps(mean, params):
    mean = mean.at[IDX_WPRIME].set(jnp.maximum(mean[IDX_WPRIME], 0.0))
    mean = mean.at[IDX_VO2].set(jnp.maximum(mean[IDX_VO2], 0.0))
    ...
    return mean  # covariance NEVER touched
```
After clamping, the posterior `GaussianState(mean=clamped_mean, cov=original_cov)` is incoherent: the mean is at the feasible boundary (e.g., W'_bal = 0) while the covariance is centered on the pre-clamped infeasible mean. The implied 1-σ confidence ellipsoid extends into the forbidden region.

**Consequence:** UKF sigma points on the next prediction step are drawn from an incoherent Gaussian. The filter accumulates bias toward constraint boundaries.

**Fix required (await OK):** Replace naive mean clamp with a constrained moment update. Options:
- Projected Gaussian: reflect the covariance using the constraint gradient (truncated-normal moment matching for each scalar constraint)
- Or accept the approximation but symmetrise: clamp mean AND project covariance (zero the cross-terms that would push into the infeasible region)

This is a non-trivial mathematical change — propose exact fix with reference before implementing.

---

#### B-06 — `app/control/nmpc_engine.py` uses 9-state architecture
**Severity:** HIGH — state-dimension mismatch; produces wrong trajectory planning  
**Location:** `app/control/nmpc_engine.py`, top-level  
**Detail:**
```python
from app.engine.observation.aerobic_observer import (
    AerobicObserverParams, DEFAULT_OBSERVER_PARAMS
)
# Uses DEFAULT_OBSERVER_PARAMS.RMSSD_ref, .k_rmssd, .HR_intr
# These are 9-state observer parameters
# The cardio slice uses CardioObsParams with RMSSD = RMSSD_ref × AT
```
The NMPC planner models `[V_vagal, W_prime_bal, load_acute, load_chronic]` — mixing a 9-state component (`V_vagal`) with a 6-state concept (`W_prime_bal`). The RMSSD model is the exponential version from the 9-state observer, not the linear AT formulation in the 6-state slice.

**Fix required (await OK):** Reconcile NMPC state vector to align with the canonical 6-state architecture. Propose rewritten planning model.

---

#### B-07 — `app/inference/nlme_model.py`: wrong sample key in `get_individual_theta`
**Severity:** MEDIUM — silent wrong value (no exception)  
**Location:** `nlme_model.py`, `get_individual_theta()`  
**Detail:**
```python
scale_eta = posterior_samples.get("scale_eta_log", ...)
# NumPyro samples "scale_eta" (HalfNormal), not "scale_eta_log"
```
When the key is not found, `get` returns the default, so all individual scale parameters default to their fallback — meaning the inter-individual variance Ω is ignored during personalisation. Athletes all get the population mean, not their individualised posterior. Violates I2 (everything probabilistic).

---

### TIER 3 — STRUCTURAL DEBT (two architectures; no orchestrator)

#### B-08 — Dual architecture: 9-state vs 6-state, no canonical orchestrator
**Severity:** HIGH — blocks end-to-end operation  
**Detail:** The repository contains two complete but mutually incompatible aerobic architectures:

**OLD (9-state):**
- `app/engine/observation/aerobic_observer.py` — STATE_DIM=9: [NE, E, V_vagal, P_a, PaCO2, PbCO2, SpO2, V_E, W_prime_bal]
- `app/engine/assimilation/ukf_filter.py` — stub, only `GaussianState`
- `app/inference/nlme_model.py` — AerobicNLME for 9-state (broken import)
- Referenced by: `aerobic_pipeline.py`, `safety_filter.py`, `lexical_filter.py`, `nmpc_engine.py`

**NEW (6-state):**
- `app/slices/cardiorespiratory/` — self-contained ODE+filter+NLME+obs+envelope
- Referenced by: 3 tests only

No canonical orchestrator wires the 6-state slice end-to-end. `aerobic_pipeline.py` claims to be the orchestrator but is DEPRECATED and broken.

**Fix required (await OK):** Create `app/slices/cardiorespiratory/orchestrator.py` wiring the 6-state pipeline; archive (not delete) the 9-state modules.

---

#### B-09 — Gate Zero has no implementation
**Severity:** CRITICAL — campaign goal cannot be evaluated  
**Detail:** `app/validation/` does not exist. `backtest_engine.py` does not exist. Walk-forward out-of-sample validation (twin must beat 7-day rolling mean MAE in ≥60% of users at 60 days) has zero lines of code.

This is the production promotion criterion. Nothing can be promoted to Gate Zero without it.

**Fix required (await OK):** Implement `app/validation/backtest_engine.py` with:
- Walk-forward splitter (60-day window)
- 7-day rolling mean baseline MAE computation
- Twin MAE computation using `CardioStateFilter.update_state` with historical observations
- `GateZeroDecision` enum: `PASS | FAIL | INSUFFICIENT_DATA`
- Per-user and aggregate statistics

---

### TIER 4 — SILENT FAILURE RISK (violates I1: fail-loud)

#### B-10 — `envelope.py` swallows `build_engine_priors` exceptions silently
**Severity:** MEDIUM — violates I1 (fail-loud, never fail-silent)  
**Location:** `app/slices/cardiorespiratory/envelope.py`, `build_cardiorespiratory_envelope()`  
**Detail:**
```python
try:
    engine_priors = build_engine_priors(athlete_data, genotype)
except Exception:
    pass  # silently falls through — priors NOT applied
```
If `build_engine_priors` fails for any reason, the envelope is built without genotype-adjusted priors. The function returns a `Phase3Envelope` that looks valid but has wrong/missing priors. No log, no status flag, no exception. Violates I1.

**Fix required (await OK):** Re-raise or log + set a degraded status. The orchestrator's `ok|degraded|failed` framework should propagate this.

---

### TIER 5 — STYLE / DEPENDENCY HYGIENE

#### B-11 — `requirements.txt`: missing critical dependencies, no version pins
**Severity:** MEDIUM  
**Detail:**

| Missing package | Required by |
|----------------|-------------|
| `do-mpc` | `app/control/nmpc_engine.py` — `AerobicNMPC` cannot be instantiated without it |
| `pydantic` | `app/schemas/`, `app/compliance/lexical_filter.py` |
| `pytest` | All tests |
| `hypothesis` | Property-based tests (if wired per HLD) |
| `ruff` | Code quality gate |
| `mypy` | Type-checking gate |
| `structlog` | Structured logging (referenced in HLD) |
| `evidently` / `nannyml` | Drift detection (referenced in HLD) |
| `mlflow` / `dvc` | Experiment tracking |
| `polars` | Data processing (referenced in HLD) |
| `botorch` / `pyPESTO` | Bayesian optimisation / parameter estimation |
| `particles` | SMC inference (referenced in HLD) |

**No version pins on any dependency.** JAX/diffrax/equinox/NumPyro are known to have breaking API changes between minor versions. This is a reproducibility and CI stability risk.

---

## SECTION 3 — WHAT IS MISSING FOR THE AEROBIC SLICE TO RUN END-TO-END

The following must exist before the aerobic slice can execute a single step of prediction + assimilation:

| # | What is missing | Blocking which gate |
|---|-----------------|-------------------|
| M-01 | `app/slices/cardiorespiratory/orchestrator.py` — canonical 6-state pipeline orchestrator | All phases |
| M-02 | Fixed `_apply_physical_clamps` with Gaussian-coherent moment update | Phase 2 (constrained filter) |
| M-03 | Fixed `safety_filter.py` imports + correct 6-state indices | Phase 5 (control) |
| M-04 | Fixed `lexical_filter.py` import (no `AerobicTransitionParams`) | Every output |
| M-05 | Fixed `nlme_model.py` import + `get_individual_theta` key | Phase 4 (NLME personalisation) |
| M-06 | Fixed `nmpc_engine.py` state-dimension alignment to 6-state | Phase 5 (control) |
| M-07 | `app/validation/backtest_engine.py` with walk-forward harness | Gate Zero (Phase 3) |
| M-08 | `do-mpc` + version-pinned `requirements.txt` | Phase 5 (control) |
| M-09 | Archive `aerobic_pipeline.py` and 9-state modules from active import path | Phase 0 (hygiene) |
| M-10 | Fail-loud `except Exception: pass` → re-raise in `envelope.py` | I1 (always) |

---

## SECTION 4 — CAMPAIGN PHASE CHECKLIST

Per the Execution Plan (22-step Compass):

### Phase 0 — Hygiene
- [ ] Archive `app/slices/aerobic_pipeline.py` (marked DEPRECATED; 4 broken imports)
- [ ] Archive / isolate 9-state modules from active import path
- [ ] Elect `app/slices/cardiorespiratory/orchestrator.py` as canonical L2→L8 wiring
- [ ] Fix broken imports: `safety_filter.py`, `lexical_filter.py`, `nlme_model.py`
- [ ] Add `do-mpc` + missing packages to `requirements.txt`; add version pins

### Phase 1 — Fail-Loud Spine
- [ ] `app/engine/orchestrator.py` with `ok|degraded|failed` per module
- [ ] Orchestrator REFUSES to emit guidance if nuclear module (safety, ODE) failed
- [ ] Test: inject module failure → verify no valid-looking output emitted
- [ ] Fix `except Exception: pass` in `envelope.py` → propagate to orchestrator status

### Phase 2 — Constrained Filter
- [ ] Replace `_apply_physical_clamps` naive mean-clamp with Gaussian-coherent update
- [ ] Proposal with mathematical reference before implementing (await OK)
- [ ] Verify: UKF covariance remains PSD after clamping
- [ ] Test: 100-step UKF with exhaustion → verify covariance coherence at W'=0

### Phase 3 — Gate Zero Validation
- [ ] Implement `app/validation/backtest_engine.py`
- [ ] Walk-forward splitter: 60-day window
- [ ] 7-day rolling mean baseline MAE
- [ ] Twin MAE using `CardioStateFilter.update_state` on historical data
- [ ] `GateZeroDecision`: PASS | FAIL | INSUFFICIENT_DATA
- [ ] Gate criterion: twin beats rolling mean in ≥60% of users at 60 days

### Phase 4 — NLME + Identifiability
- [ ] FIM / profile likelihood analysis on D=2 parameters (VO2_max_baseline, W_prime_capacity)
- [ ] Fix `get_individual_theta` wrong sample key (`scale_eta_log` → `scale_eta`)
- [ ] Wire NumPyro posterior to `CardioTransitionParams` personalisation
- [ ] Non-identifiable parameters: fix at prior mean, mark `FIXED_AT_PRIOR`

### Phase 5 — Control
- [ ] Reconcile `nmpc_engine.py` state vector to 6-state architecture
- [ ] Fix `safety_filter.py` symbol imports + state indices
- [ ] PSF: verify 1-step projection uses 6-state UKF, correct AT and W'_bal indices
- [ ] ACWR chance constraints: verify k=2.33 (α=0.01, Gabbett 2016)

---

## SECTION 5 — INVARIANT COMPLIANCE CHECK

| Invariant | Status | Evidence |
|-----------|--------|----------|
| **I1** Fail-loud | PARTIAL | Fail-loud: ODE NaN guards ✓; UKF NaN raise ✓; NLME ELBO NaN raise ✓. Fail-SILENT: `envelope.py` `except: pass` ✗; `safety_filter.py` ImportError means no safety at all ✗ |
| **I2** Probabilistic | PARTIAL | CardioNLME: non-centred parametrisation ✓; NLME SVI ✓. But `get_individual_theta` returns wrong values silently ✗; UKF clamp breaks Gaussian coherence ✗ |
| **I3** No magic constants | PASS | `CardioSliceParams` cites Boillet 2024, Clarke-Skiba 2013, Coyle 2001, Dempsey 2006, Arai 1989, Kontro 2026. `priors.py` cites NIST. |
| **I4** Safety as hard constraint | PARTIAL | Hard constraints in `Phase3Envelope` ✓. But PSF module imports fail (no safety layer active) ✗ |
| **I5** Math continuity | PASS | ODE uses `jnp.where`, `jnp.maximum`, `jnp.clip`. No Python if/then on traced values. Tsit5 solver ✓ |
| **I6** Epistemic honesty | PARTIAL | `CardioNLME`: D=2 personalised, others fixed at prior ✓. But: `get_individual_theta` wrong key = silent misidentification ✗ |

---

## SECTION 6 — PATH A COMPLIANCE NOTE

`app/engine/priors.py` exports `BAYESIAN_PRIOR_MAPPINGS` with key `"DunedinPACE"` mapped to `"p3_biological_age_rate_prior"`. The `lexical_filter.py` forbidden patterns include `biological.?age`. This key should never surface in user-facing output (it is engine-internal), but if it leaks through any template it will trigger a Path A violation. Flag for verification when orchestrator is wired.

The `_FORBIDDEN` patterns in `lexical_filter.py` correctly cover all required terms: `diagnos*`, `treat*`, `therap*`, `cure*`, `prescri*`, `medic*`, `diseas*`, `longevity`, `lifespan`, `biological.?age`, etc.

---

## SECTION 7 — FILE MAP SUMMARY

```
BROKEN (ImportError at load):
  app/slices/aerobic_pipeline.py          — DEPRECATED; 4 broken imports
  app/control/safety_filter.py            — 3 broken imports + wrong state indices
  app/compliance/lexical_filter.py        — 1 broken import
  app/inference/nlme_model.py             — 1 broken import + silent wrong value

MISSING (does not exist):
  app/validation/                         — Gate Zero has no code
  app/engine/solvers/                     — referenced by aerobic_pipeline.py
  app/slices/cardiorespiratory/orchestrator.py  — no canonical end-to-end wiring

MATH BUG (runs but wrong):
  app/slices/cardiorespiratory/filter.py  — _apply_physical_clamps breaks Gaussian coherence

SILENT FAILURE (runs but wrong):
  app/inference/nlme_model.py             — get_individual_theta returns wrong personalisation
  app/slices/cardiorespiratory/envelope.py — except: pass swallows engine_priors failure

CLEAN (implemented correctly):
  app/slices/cardiorespiratory/ode.py
  app/slices/cardiorespiratory/observation.py
  app/slices/cardiorespiratory/nlme.py    (minus get_individual_theta)
  app/engine/priors.py
  app/engine/phase3_envelope.py
  app/engine/unit_converter.py
  app/engine/assimilation/ukf_filter.py  (only GaussianState — correct)
  tests/test_cardiorespiratory_slice.py   (physics/stability only — not Gate Zero)
```

---

## SECTION 8 — RECOMMENDED EXECUTION ORDER (Phase 0 first)

Before any write action, awaiting your **explicit OK**. Proposed order:

1. **Phase 0-A** — Fix the 4 broken imports (safe, mechanical, no physics change):
   - `safety_filter.py`: remove broken imports; add correct 6-state symbol references as stubs
   - `lexical_filter.py`: remove `AerobicTransitionParams` import
   - `nlme_model.py`: remove `X0_DEFAULT` import; fix `get_individual_theta` key
   - `aerobic_pipeline.py`: move to `_deprecated/` folder (not delete)

2. **Phase 0-B** — `requirements.txt`: add `do-mpc`; add version pins for JAX stack

3. **Phase 0-C** — Fix `except Exception: pass` in `envelope.py` → `except Exception as exc: raise RuntimeError(...) from exc`

4. **Phase 1** — Create `app/slices/cardiorespiratory/orchestrator.py` with ok|degraded|failed status

5. **Phase 2** — Propose Gaussian-coherent clamping fix (mathematical proposal first, await OK)

6. **Phase 3** — Implement Gate Zero validation harness

7. **Phase 4** — Fix NLME personalisation; FIM analysis

8. **Phase 5** — Reconcile NMPC + PSF to 6-state architecture

---

*End of AUDIT_REPORT.md. Awaiting explicit OK before proceeding to Phase 0 actions.*
