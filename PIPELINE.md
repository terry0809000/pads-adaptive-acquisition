# Execution interfaces — Adaptive versus fixed smartwatch task acquisition

This guide describes the preserved source interfaces by reading the code. No research module was imported, no command was run, and no performance or portability validation is implied. Use the original project instructions for the scientific run order.

## Data → analysis → outputs

**Input contract:** PADS 1.0.0 from PhysioNet, CC BY-NC-SA 4.0. Acquire source recordings separately.

1. Acquire inputs from their original providers and configure the paths defined in the source.
2. Install the recorded dependencies in an isolated environment.
3. Preserve grouping, seeds, development/evaluation separation and any protocol/hash gates.
4. Run acquisition/extraction and preprocessing before model development. Run evaluation only after the selection/freeze required by the original implementation.
5. Keep raw data, row-level derivatives, predictions and fitted models outside the public repository.

## Script reference

### `pads_analysis/acquisition_rules.py`

[Open source](pads_analysis/acquisition_rules.py) · Supporting module

Rules to reuse when the predictive and task-selection models are fitted. These functions do not fit an acquisition policy or claim clinical validity. Only training-fold out-of-sample predictions may be used to set thresholds.

### `pads_analysis/evaluate_primary.py`

[Open source](pads_analysis/evaluate_primary.py) · Executable entry point

Compare held-out adaptive and optimized fixed protocols on the full cohort. Required inputs are genuine out-of-fold predictions and a separately frozen cohort/split manifest. This program does not generate or fit predictions.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--cohort` | Yes |  |
| `--predictions` | Yes |  |
| `--output` | No / positional | default: Path("primary_comparison.json") |
| `--bootstrap-draws` | No / positional | default: 5000 |

### `pads_analysis/make_figures.py`

[Open source](pads_analysis/make_figures.py) · Supporting module

Publication figures drawn only from the saved held-out evaluation results.

### `pads_analysis/reviewer_round1/make_round1.py`

[Open source](pads_analysis/reviewer_round1/make_round1.py) · Supporting module

Exploratory reviewer plots; no model training or threshold modification.

### `pads_analysis/reviewer_round2/package_round2.py`

[Open source](pads_analysis/reviewer_round2/package_round2.py) · Supporting module

Package the second-round result snapshot with its reproducibility inputs.

### `pads_analysis/reviewer_round2/run_repeats.py`

[Open source](pads_analysis/reviewer_round2/run_repeats.py) · Executable entry point

Re-run the frozen PADS procedure on five newly specified outer splits.

### `pads_analysis/reviewer_round2/summarize_repeats.py`

[Open source](pads_analysis/reviewer_round2/summarize_repeats.py) · Supporting module

Validate complete repeated development and report split sensitivity.

### `pads_analysis/reviewer_round3/diagnostic_uncertainty.py`

[Open source](pads_analysis/reviewer_round3/diagnostic_uncertainty.py) · Supporting module

Conditional bootstrap intervals for descriptive calibration coefficients.

### `pads_analysis/reviewer_round3/make_figures.py`

[Open source](pads_analysis/reviewer_round3/make_figures.py) · Supporting module

Publication figures from saved outer-held-out round-3 results.

### `pads_analysis/reviewer_round3/package_round3.py`

[Open source](pads_analysis/reviewer_round3/package_round3.py) · Supporting module

Validate and package a cumulative, source-traceable result snapshot.

### `pads_analysis/reviewer_round3/run_round3.py`

[Open source](pads_analysis/reviewer_round3/run_round3.py) · Supporting module

Outer-held-out evaluation of probability calibration and rejection targets.

### `pads_analysis/reviewer_round4/allocation_rules.py`

[Open source](pads_analysis/reviewer_round4/allocation_rules.py) · Supporting module

Two-cutoff rejection allocation; inference never receives reference labels.

### `pads_analysis/reviewer_round4/make_figures.py`

[Open source](pads_analysis/reviewer_round4/make_figures.py) · Supporting module

Round-4 scientific figures; all values read from saved evaluation tables.

### `pads_analysis/reviewer_round4/package_round4.py`

[Open source](pads_analysis/reviewer_round4/package_round4.py) · Supporting module

Validate inference and saved outputs, then package the cumulative snapshot.

### `pads_analysis/reviewer_round4/run_round4.py`

[Open source](pads_analysis/reviewer_round4/run_round4.py) · Supporting module

Develop and evaluate diagnosis-balanced rejection allocations.

### `pads_analysis/reviewer_round5/make_figures.py`

[Open source](pads_analysis/reviewer_round5/make_figures.py) · Executable entry point

Publication figures generated directly from the saved numerical tables.

### `pads_analysis/reviewer_round5/package_round5.py`

[Open source](pads_analysis/reviewer_round5/package_round5.py) · Executable entry point

Create a cumulative archive while preserving every preceding member byte.

### `pads_analysis/reviewer_round5/random_controls.py`

[Open source](pads_analysis/reviewer_round5/random_controls.py) · Supporting module

Label-free acquisition distributions and weighted development rules.

### `pads_analysis/reviewer_round5/run_round5.py`

[Open source](pads_analysis/reviewer_round5/run_round5.py) · Executable entry point

Rebuild middle fits and evaluate random controls on the six saved partitions.

### `pads_analysis/reviewer_round5/summarize_round5.py`

[Open source](pads_analysis/reviewer_round5/summarize_round5.py) · Executable entry point

Conditional paired uncertainty and whole-cohort randomization realizations.

### `pads_analysis/reviewer_round6/make_integrated_figures.py`

[Open source](pads_analysis/reviewer_round6/make_integrated_figures.py) · Supporting module

Reproducible scientific figures from archived participant-derived tables.

### `pads_analysis/reviewer_round6/package_round6.py`

[Open source](pads_analysis/reviewer_round6/package_round6.py) · Executable entry point

Assemble the cumulative archive without altering any earlier-round entry.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--previous-archive` | Yes |  |
| `--staged-dir` | Yes |  |
| `--output` | Yes |  |

### `pads_analysis/reviewer_round6/parameterized_analysis.py`

[Open source](pads_analysis/reviewer_round6/parameterized_analysis.py) · Executable entry point

Nested participant-level evaluation of adaptive PADS task acquisition.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--features-dir` | Yes |  |
| `--output` | Yes |  |
| `--without-nms` | No / positional |  |
| `--outer-fold` | No / positional |  |

### `pads_analysis/reviewer_round6/run_sensitivity.py`

[Open source](pads_analysis/reviewer_round6/run_sensitivity.py) · Executable entry point

Fixed, exploratory one-factor sensitivities; preserve original primary outputs.

### `pads_analysis/reviewer_round6/summarize_sensitivity.py`

[Open source](pads_analysis/reviewer_round6/summarize_sensitivity.py) · Executable entry point

### `pads_analysis/reviewer_round6/validate_integration.py`

[Open source](pads_analysis/reviewer_round6/validate_integration.py) · Supporting module

Check reported effect sizes, denominators, citations and deliverable structure.

### `pads_analysis/reviewer_round7/assemble_round7.py`

[Open source](pads_analysis/reviewer_round7/assemble_round7.py) · Executable entry point

Integrate real complete-refit results with the preserved primary evidence.

### `pads_analysis/reviewer_round7/build_reference_audit.py`

[Open source](pads_analysis/reviewer_round7/build_reference_audit.py) · Executable entry point

Create claim-specific citation records without redistributing source articles.

### `pads_analysis/reviewer_round7/build_review_audit.py`

[Open source](pads_analysis/reviewer_round7/build_review_audit.py) · Executable entry point

Twenty source-informed research questions, with completed experimental answers.

### `pads_analysis/reviewer_round7/checkpoint.py`

[Open source](pads_analysis/reviewer_round7/checkpoint.py) · Executable entry point

Snapshot completed refits without touching a running worker's files.

### `pads_analysis/reviewer_round7/finish_analysis.py`

[Open source](pads_analysis/reviewer_round7/finish_analysis.py) · Executable entry point

Run result-dependent integration only after the frozen computation completes.

### `pads_analysis/reviewer_round7/grouped_analysis.py`

[Open source](pads_analysis/reviewer_round7/grouped_analysis.py) · Executable entry point

Round 7 copy of the original model; grouped identity splits protect bootstrap copies. The original run_outer/main remain archived for comparison and are not used here. Use run_bootstrap.py for this round.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--features-dir` | Yes |  |
| `--output` | Yes |  |
| `--without-nms` | No / positional |  |
| `--outer-fold` | No / positional |  |

### `pads_analysis/reviewer_round7/methods_revisions.py`

[Open source](pads_analysis/reviewer_round7/methods_revisions.py) · Supporting module

Methods edited for scientific argument and continuity; primary settings unchanged.

### `pads_analysis/reviewer_round7/package_round7.py`

[Open source](pads_analysis/reviewer_round7/package_round7.py) · Executable entry point

Assemble the cumulative archive without altering any earlier-round entry.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--previous-archive` | Yes |  |
| `--staged-dir` | Yes |  |
| `--output` | Yes |  |

### `pads_analysis/reviewer_round7/plot_bootstrap.py`

[Open source](pads_analysis/reviewer_round7/plot_bootstrap.py) · Executable entry point

Publication figures from the complete audited 500-replicate result tables.

### `pads_analysis/reviewer_round7/run_bootstrap.py`

[Open source](pads_analysis/reviewer_round7/run_bootstrap.py) · Executable entry point

Run the frozen, identity-grouped bootstrap of the complete PADS procedure.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--workers` | No / positional | default: 6 |
| `--start` | No / positional | default: 0 |
| `--stop` | No / positional | default: B |
| `--reproduce-only` | No / positional |  |

### `pads_analysis/reviewer_round7/summarize_bootstrap.py`

[Open source](pads_analysis/reviewer_round7/summarize_bootstrap.py) · Executable entry point

Audit all 500 refits and summarize empirical stability, not external validation.

### `pads_analysis/reviewer_round7/validate_integration.py`

[Open source](pads_analysis/reviewer_round7/validate_integration.py) · Executable entry point

Check numerical provenance, citation continuity and editable document structure.

### `pads_analysis/run_analysis.py`

[Open source](pads_analysis/run_analysis.py) · Executable entry point

Nested participant-level evaluation of adaptive PADS task acquisition.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--features-dir` | Yes |  |
| `--output` | Yes |  |
| `--without-nms` | No / positional |  |
| `--outer-fold` | No / positional |  |

### `pads_analysis/summarize_results.py`

[Open source](pads_analysis/summarize_results.py) · Executable entry point

Validate paired outputs and summarize the prespecified analyses.

### `pads_analysis/verify_analysis_rules.py`

[Open source](pads_analysis/verify_analysis_rules.py) · Executable entry point

Focused synthetic verification; no clinical performance is measured here.

### `pads_features_received/PADS_features_return/extract_pads_features.py`

[Open source](pads_features_received/PADS_features_return/extract_pads_features.py) · Executable entry point

Create a small, auditable PADS feature return package from the local raw ZIP. No model is fitted, no outcome-based feature selection is performed, and no motion-time budget or final validation split is selected by this program.

| Argument | Required | Default / allowed values |
|---|---|---|
| `--archive` | Yes |  |
| `--output` | No / positional | default: Path("PADS_features_return") |

### `prepare_folds.py`

[Open source](prepare_folds.py) · Supporting module

| Argument | Required | Default / allowed values |
|---|---|---|
| `--features-dir` | No / positional | default: Path('pads_features_received/PADS_features_return') |

## Reproduction limits

Absolute paths, frozen-file references and platform-specific assumptions are preserved. Configure a matching local workspace rather than bypassing scientific integrity guards. A repository containing existing scripts is not evidence that every data or environment prerequisite is satisfied. No original clinical results or completed-run artifacts are redistributed.
