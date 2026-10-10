# Adaptive versus fixed smartwatch task acquisition

**AI for Health · Research pipeline · Source-code release**

A wearable has limited time to ask useful questions. This project compares adaptive and fixed task acquisition using PADS smartwatch recordings, exploring how the next measurement is chosen.

This repository preserves the saved scientific implementation, with its original execution notes and a source-specific pipeline guide.

## Pipeline

1. **Acquire inputs.** PADS 1.0.0 from PhysioNet, CC BY-NC-SA 4.0. Acquire source recordings separately.
2. **Prepare the environment.** Use the project-specific dependency files listed below in an isolated environment. Preserve the versions and seeds recorded in the source release.
3. **Prepare data and run the model or analysis.** Follow the original execution notes and entry points below, preserving time order, participant grouping and fixed evaluation boundaries.
4. **Evaluate and summarize.** Use the included evaluation/analysis scripts. Write generated data, predictions, fitted objects and caches to a private working copy.

This is a source-only repository. Raw data, individual-level derivatives and fitted checkpoints are not bundled. Some saved-evidence reproduction commands in the original notes require artifacts from the original source release; obtain those independently where authorized.

## Environment

- [`pads_analysis/reviewer_round7/ENVIRONMENT.json`](pads_analysis/reviewer_round7/ENVIRONMENT.json)
- [`pads_features_received/PADS_features_return/requirements.txt`](pads_features_received/PADS_features_return/requirements.txt)
- [`requirements_analysis.txt`](requirements_analysis.txt)

## Execution

See the [pipeline and command reference](PIPELINE.md) for entry points, required arguments and the source-specific input contract.

Run commands from the repository root unless the original notes specify another working directory. Use a fresh copy and configure local data paths before running.

### Original execution notes

- [README.md](docs/original/README.md)

### Scientific source inventory

<details>
<summary>Browse the preserved scientific implementation</summary>

- [`pads_analysis/acquisition_rules.py`](pads_analysis/acquisition_rules.py)
- [`pads_analysis/evaluate_primary.py`](pads_analysis/evaluate_primary.py)
- [`pads_analysis/make_figures.py`](pads_analysis/make_figures.py)
- [`pads_analysis/reviewer_round1/make_round1.py`](pads_analysis/reviewer_round1/make_round1.py)
- [`pads_analysis/reviewer_round1/package_round1.py`](pads_analysis/reviewer_round1/package_round1.py)
- [`pads_analysis/reviewer_round2/build_round2_report.py`](pads_analysis/reviewer_round2/build_round2_report.py)
- [`pads_analysis/reviewer_round2/package_round2.py`](pads_analysis/reviewer_round2/package_round2.py)
- [`pads_analysis/reviewer_round2/run_repeats.py`](pads_analysis/reviewer_round2/run_repeats.py)
- [`pads_analysis/reviewer_round2/summarize_repeats.py`](pads_analysis/reviewer_round2/summarize_repeats.py)
- [`pads_analysis/reviewer_round3/diagnostic_uncertainty.py`](pads_analysis/reviewer_round3/diagnostic_uncertainty.py)
- [`pads_analysis/reviewer_round3/make_figures.py`](pads_analysis/reviewer_round3/make_figures.py)
- [`pads_analysis/reviewer_round3/package_round3.py`](pads_analysis/reviewer_round3/package_round3.py)
- [`pads_analysis/reviewer_round3/run_round3.py`](pads_analysis/reviewer_round3/run_round3.py)
- [`pads_analysis/reviewer_round4/allocation_rules.py`](pads_analysis/reviewer_round4/allocation_rules.py)
- [`pads_analysis/reviewer_round4/make_figures.py`](pads_analysis/reviewer_round4/make_figures.py)
- [`pads_analysis/reviewer_round4/package_round4.py`](pads_analysis/reviewer_round4/package_round4.py)
- [`pads_analysis/reviewer_round4/run_round4.py`](pads_analysis/reviewer_round4/run_round4.py)
- [`pads_analysis/reviewer_round5/make_figures.py`](pads_analysis/reviewer_round5/make_figures.py)
- [`pads_analysis/reviewer_round5/package_round5.py`](pads_analysis/reviewer_round5/package_round5.py)
- [`pads_analysis/reviewer_round5/random_controls.py`](pads_analysis/reviewer_round5/random_controls.py)
- [`pads_analysis/reviewer_round5/run_round5.py`](pads_analysis/reviewer_round5/run_round5.py)
- [`pads_analysis/reviewer_round5/summarize_round5.py`](pads_analysis/reviewer_round5/summarize_round5.py)
- [`pads_analysis/reviewer_round6/build_review_audit.py`](pads_analysis/reviewer_round6/build_review_audit.py)
- [`pads_analysis/reviewer_round6/make_integrated_figures.py`](pads_analysis/reviewer_round6/make_integrated_figures.py)
- [`pads_analysis/reviewer_round6/package_round6.py`](pads_analysis/reviewer_round6/package_round6.py)
- [`pads_analysis/reviewer_round6/parameterized_analysis.py`](pads_analysis/reviewer_round6/parameterized_analysis.py)
- [`pads_analysis/reviewer_round6/run_sensitivity.py`](pads_analysis/reviewer_round6/run_sensitivity.py)
- [`pads_analysis/reviewer_round6/summarize_sensitivity.py`](pads_analysis/reviewer_round6/summarize_sensitivity.py)
- [`pads_analysis/reviewer_round6/validate_integration.py`](pads_analysis/reviewer_round6/validate_integration.py)
- [`pads_analysis/reviewer_round7/assemble_round7.py`](pads_analysis/reviewer_round7/assemble_round7.py)
- [`pads_analysis/reviewer_round7/build_reference_audit.py`](pads_analysis/reviewer_round7/build_reference_audit.py)
- [`pads_analysis/reviewer_round7/build_review_audit.py`](pads_analysis/reviewer_round7/build_review_audit.py)
- [`pads_analysis/reviewer_round7/checkpoint.py`](pads_analysis/reviewer_round7/checkpoint.py)
- [`pads_analysis/reviewer_round7/finish_analysis.py`](pads_analysis/reviewer_round7/finish_analysis.py)
- [`pads_analysis/reviewer_round7/grouped_analysis.py`](pads_analysis/reviewer_round7/grouped_analysis.py)
- [`pads_analysis/reviewer_round7/methods_revisions.py`](pads_analysis/reviewer_round7/methods_revisions.py)
- [`pads_analysis/reviewer_round7/package_round7.py`](pads_analysis/reviewer_round7/package_round7.py)
- [`pads_analysis/reviewer_round7/plot_bootstrap.py`](pads_analysis/reviewer_round7/plot_bootstrap.py)
- [`pads_analysis/reviewer_round7/run_bootstrap.py`](pads_analysis/reviewer_round7/run_bootstrap.py)
- [`pads_analysis/reviewer_round7/summarize_bootstrap.py`](pads_analysis/reviewer_round7/summarize_bootstrap.py)
- [`pads_analysis/reviewer_round7/validate_integration.py`](pads_analysis/reviewer_round7/validate_integration.py)
- [`pads_analysis/run_analysis.py`](pads_analysis/run_analysis.py)
- [`pads_analysis/summarize_results.py`](pads_analysis/summarize_results.py)
- [`pads_analysis/verify_analysis_rules.py`](pads_analysis/verify_analysis_rules.py)
- [`pads_features_received/PADS_features_return/extract_pads_features.py`](pads_features_received/PADS_features_return/extract_pads_features.py)
- [`prepare_folds.py`](prepare_folds.py)
- [`prepare_random_input_manifest.py`](prepare_random_input_manifest.py)

</details>

## Repository contents and rights

Only scientific code, dependencies, configuration and execution documentation are included. Manuscripts, submission documents, raw participant records, trained models, credentials, caches and software environments are excluded. Notebook cell outputs and execution counters are cleared for publication; code cells are retained.

Existing source and third-party license notices govern their respective files. No blanket license is invented for material whose original release did not specify one. Original dataset licenses and access requirements continue to apply.

Research software is not a clinical deployment claim. Negative findings and the source release's interpretation limits remain applicable. No tests or experiments were run during this upload, as requested by the author.
