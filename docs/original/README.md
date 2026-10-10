# Supplementary File S2 — analysis code and aggregate supporting data

This supplement supports the PADS comparison of adaptive, optimized fixed, fixed-sequence stopping and random next-task acquisition. It preserves the original analysis code. No model was retrained when preparing this journal package.

## Inputs and rights

Obtain PADS version 1.0.0 from https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/ (DOI 10.13026/m0w9-zx22). The source data use CC BY-NC-SA 4.0. This supplement does not relicense or redistribute the source recordings or participant-level data. Cite the source article, its correction where relevant, the dataset and PhysioNet as in the manuscript.

The source feature extraction occurred under Python 3.14.2, NumPy 2.4.6 and SciPy 1.17.1. Predictive analyses used Python 3.12 with the versions listed in requirements_analysis.txt. Install extraction and analysis environments separately if exact numerical reproduction is required. Unpinned plotting-only package versions do not alter the frozen analysis definitions; the archived environment record records the original runtime where available.

## Reproduction sequence

Run commands from the extracted supplement root, replacing the raw archive path with the downloaded PADS ZIP. Create all results in a working copy. The complete analysis is computationally substantial; the 500-replicate extension alone required 10360000 logistic fits. Do not start the full run merely to inspect the aggregate results.

```bash
python pads_features_received/PADS_features_return/extract_pads_features.py --archive /path/to/PADS.zip --output pads_features_received/PADS_features_return
python prepare_folds.py
python pads_analysis/run_analysis.py --features-dir pads_features_received/PADS_features_return --output pads_analysis/results_primary
python pads_analysis/run_analysis.py --features-dir pads_features_received/PADS_features_return --output pads_analysis/results_without_nms --without-nms
python pads_analysis/summarize_results.py
python pads_analysis/reviewer_round2/run_repeats.py
python pads_analysis/reviewer_round2/summarize_repeats.py
```

The following extensions use the previously fitted systems and outputs. The calibration and class-allocation scripts also write their own summaries. The random-control script rebuilds middle systems and reuses the outer systems. Record the fresh reproduction input hashes before running it; the helper refuses to replace an existing manifest. The historical hash ledger is preserved separately and is never overwritten. Its original runtime-dependent hashes should not be represented as hashes of newly generated files.

```bash
python pads_analysis/reviewer_round3/run_round3.py
python pads_analysis/reviewer_round3/diagnostic_uncertainty.py
python pads_analysis/reviewer_round4/run_round4.py
python prepare_random_input_manifest.py
python pads_analysis/reviewer_round5/run_round5.py
python pads_analysis/reviewer_round5/summarize_round5.py
python pads_analysis/reviewer_round6/run_sensitivity.py
python pads_analysis/reviewer_round6/summarize_sensitivity.py
```

The full-refit extension generates a new checksum ledger for the reproduction directory, first checks an original-fold reproduction and then runs 500 complete resamples:

```bash
python pads_analysis/reviewer_round7/run_bootstrap.py --workers 6 --start 0 --stop 500
python pads_analysis/reviewer_round7/summarize_bootstrap.py
```

The original-fold reproduction gate must pass before the complete-refit run. Preserve checksums and sample identities; do not remove a guard to force agreement. Numerical differences can arise if feature extraction is repeated in a different environment. The received feature payload hash and this limitation are reported in Supplementary Methods S8. The journal preparation did not rerun extraction from the raw archive or all 500 refits.

## What is directly verifiable without refitting

The tables directories contain cohort-level and fold/replicate-level summaries, paired contrasts, interval summaries, calibration and randomization aggregates, and full-refit sample-size summaries. These permit independent checking of the reported aggregate effects and figure values. The original participant-level feature cache, stored fitted systems and complete intermediate prediction/identity records are preserved separately in the previously delivered author archive PADS_round7_integrated_package.zip. That historical archive remains a separate author record. It is not included in this submission, whose combined uploaded files must remain below the Sensors 120 MB limit.

prepare_folds.py was checked against all 390 archived original outer-fold assignments; they agreed exactly. Code files were syntax-checked; this is not a new end-to-end model-validation experiment.

No final all-cohort clinical deployment model is supplied. Earlier conditional intervals do not incorporate refitting. The 500-replicate stability intervals are descriptive empirical intervals, not calibrated tests or optimism-corrected estimates. The later calibration and mask-permutation procedures were not rerun within the 500 participant resamples.
