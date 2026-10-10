"""Create a small, auditable PADS feature return package from the local raw ZIP.

No model is fitted, no outcome-based feature selection is performed, and no
motion-time budget or final validation split is selected by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy.signal import butter, get_window, sosfiltfilt, welch

TASKS = ["Relaxed", "RelaxedTask", "StretchHold", "LiftHold", "HoldWeight",
         "PointFinger", "DrinkGlas", "CrossArms", "TouchIndex", "TouchNose", "Entrainment"]
WRISTS = ["LeftWrist", "RightWrist"]
SENSORS = ["Accelerometer", "Gyroscope"]
BANDS = [(0.5, 3.0), (3.0, 7.0), (7.0, 12.0), (12.0, 20.0)]
ARCHIVE_HASH = "197f7e19d029fd0db8ba6897bf136b6f45c798534cac3ef0110e4bd0228d0a75"
MANIFEST_HASH = "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
CONFIG = {
    "version": "pads_motion_descriptors_v1.0",
    "created_before_any_model_fit_in_this_study": True,
    "first_seconds_discarded_from_features": 0.5,
    "raw_acquisition_cost_includes_discarded_samples": True,
    "split_if_gap_seconds_exceeds": 0.1,
    "resample_hz": 100.0,
    "interpolation": "linear within each continuous segment only; no extrapolation",
    "minimum_uniform_samples_per_segment": 200,
    "dynamic_filter": "second-order Butterworth high-pass, 0.5 Hz, forward and backward per segment",
    "welch": {"segment_samples": 200, "overlap_samples": 100, "fft_samples": 400,
              "window": "periodic Hann", "detrend": "constant", "scaling": "density",
              "average": "mean", "segment_pooling": "weighted by complete Welch window count"},
    "power_bands_hz": BANDS,
    "band_edges": "lower inclusive, upper exclusive; final band includes 20 Hz",
    "power_integration": "sum of spectral density bins times 0.25 Hz",
    "spectral_axes": "sum of three axis power spectra; no norm spectrum frequency doubling",
    "frequency_search_hz": [3.0, 12.0],
    "time_statistics": "pooled retained uniform samples; derivatives stay within segments",
    "near_zero_total_spectral_power": 1e-20,
    "no_spectral_energy_convention": "fractions, entropy, centroid, peak frequency and concentration equal zero",
    "bilateral_task_features": "mean and absolute difference of the left/right transformed descriptors",
    "quality_flags_as_predictors": False,
    "standardization_feature_selection_dimensionality_reduction": "none; reserved for training folds",
    "budget_definition": "not selected by feature extraction; pending design decision",
    "source_archive_sha256": ARCHIVE_HASH,
    "source_manifest_sha256": MANIFEST_HASH,
}

FEATURES = ["log1p_dynamic_vector_rms", "log1p_dynamic_norm_median",
            "log1p_dynamic_norm_iqr", "log1p_dynamic_norm_p95", "log1p_vector_derivative_rms"]
FEATURES += [f"log1p_power_{lo:g}_{hi:g}hz" for lo, hi in BANDS]
FEATURES += [f"fraction_power_{lo:g}_{hi:g}hz" for lo, hi in BANDS]
FEATURES += ["peak_frequency_3_12hz", "spectral_entropy_0.5_20hz",
             "peak_concentration_3_12hz", "spectral_centroid_0.5_20hz"]
TASK_FEATURES = [f"{sensor}_{aggregation}_{feature}"
                 for sensor in SENSORS for aggregation in ["bilateral_mean", "bilateral_absdiff"]
                 for feature in FEATURES]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_segments(timestamps, values):
    """Trim the start, split long gaps, then interpolate within each segment."""
    t = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or x.shape != (len(t), 6) or len(t) < 2:
        raise ValueError("Expected one time channel and six sensor channels")
    if not np.isfinite(t).all() or not np.isfinite(x).all() or not np.all(np.diff(t) > 0):
        raise ValueError("Nonfinite values or nonincreasing timestamps")
    elapsed = t - t[0]
    keep = elapsed >= CONFIG["first_seconds_discarded_from_features"]
    t, x = elapsed[keep], x[keep]
    if len(t) < 2:
        raise ValueError("No usable samples after initial trim")
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > CONFIG["split_if_gap_seconds_exceeds"]) + 1, len(t)]
    segments, spans = [], []
    short_segments = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            short_segments += 1
            continue
        tt, xx = t[start:end], x[start:end]
        n = int(np.floor((tt[-1] - tt[0]) * CONFIG["resample_hz"])) + 1
        grid = tt[0] + np.arange(n) / CONFIG["resample_hz"]
        grid = grid[grid <= tt[-1]]
        if len(grid) < CONFIG["minimum_uniform_samples_per_segment"]:
            short_segments += 1
            continue
        segments.append(np.column_stack([np.interp(grid, tt, xx[:, j]) for j in range(6)]))
        spans.append([float(grid[0]), float(grid[-1])])
    qc = {"continuous_segments_after_trim": int(len(boundaries) - 1),
          "retained_segments": len(segments), "short_segments_not_used": short_segments,
          "retained_uniform_samples": sum(len(s) for s in segments), "retained_time_spans": spans}
    return segments, qc


def sensor_descriptors(segments):
    """Seventeen rotation-invariant amplitude and spectral descriptors."""
    fs = CONFIG["resample_hz"]
    sos = butter(2, 0.5, btype="highpass", fs=fs, output="sos")
    norms, squares, differences, psds, weights = [], [], [], [], []
    for raw in segments:
        # Centering avoids finite-precision residuals from a perfectly constant signal.
        dynamic = sosfiltfilt(sos, raw - raw.mean(axis=0), axis=0)
        norms.append(np.linalg.norm(dynamic, axis=1))
        squares.append(np.sum(dynamic ** 2, axis=1))
        differences.append(np.sum((np.diff(dynamic, axis=0) * fs) ** 2, axis=1))
        f, power = welch(dynamic, fs=fs, window=get_window("hann", 200, fftbins=True),
                         nperseg=200, noverlap=100, nfft=400, detrend="constant",
                         return_onesided=True, scaling="density", axis=0, average="mean")
        psds.append(power.sum(axis=1))
        weights.append(1 + (len(dynamic) - 200) // 100)
    norm = np.concatenate(norms)
    scalar = [np.sqrt(np.mean(np.concatenate(squares))), np.median(norm),
              np.quantile(norm, 0.75) - np.quantile(norm, 0.25), np.quantile(norm, 0.95),
              np.sqrt(np.mean(np.concatenate(differences)))]
    psd = np.average(psds, axis=0, weights=weights)
    df = f[1] - f[0]
    band_power = []
    for lo, hi in BANDS:
        mask = (f >= lo) & ((f <= hi) if hi == 20 else (f < hi))
        band_power.append(float(np.sum(psd[mask]) * df))
    total = sum(band_power)
    if total <= CONFIG["near_zero_total_spectral_power"]:
        fractions = [0.0] * 4
        peak, entropy, concentration, centroid = 0.0, 0.0, 0.0, 0.0
    else:
        fractions = [v / total for v in band_power]
        broad = (f >= 0.5) & (f <= 20)
        probability = psd[broad] / psd[broad].sum()
        positive = probability > 0
        entropy = -np.sum(probability[positive] * np.log(probability[positive])) / np.log(len(probability))
        centroid = float(np.dot(f[broad], probability))
        search = (f >= 3) & (f <= 12)
        narrow_power = psd[search].sum()
        if narrow_power * df <= CONFIG["near_zero_total_spectral_power"]:
            peak, concentration = 0.0, 0.0
        else:
            peak = float(f[search][np.argmax(psd[search])])
            near_peak = search & (np.abs(f - peak) <= 0.5)
            concentration = float(psd[near_peak].sum() / narrow_power)
    result = np.r_[np.log1p(scalar), np.log1p(band_power), fractions, peak, entropy, concentration, centroid]
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("Invalid extracted descriptors")
    return result


def extract_record(t, x):
    segments, qc = record_segments(t, x)
    if not segments:
        return np.full((2, len(FEATURES)), np.nan), qc
    descriptors = np.stack([sensor_descriptors([s[:, k:k + 3] for s in segments]) for k in [0, 3]])
    return descriptors, qc


def bilateral_descriptors(values):
    # Input has shape participants x tasks x wrists x sensors x descriptors.
    left, right = values[:, :, 0], values[:, :, 1]
    out = np.stack([(left + right) / 2, np.abs(left - right)], axis=3)
    return out.reshape(values.shape[0], values.shape[1], -1)


class VerifiedArchive:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        if sha256_file(self.path) != ARCHIVE_HASH:
            raise ValueError("Raw ZIP hash differs from the successfully audited PADS archive; inspect provenance before continuing")
        self.z = zipfile.ZipFile(self.path)
        names = self.z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        manifests = [n for n in names if n.endswith("preprocessed/file_list.csv")]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one PADS participant manifest")
        self.prefix = manifests[0][:-len("preprocessed/file_list.csv")]
        raw = self.z.read(self.prefix + "SHA256SUMS.txt")
        if hashlib.sha256(raw).hexdigest() != MANIFEST_HASH:
            raise ValueError("Embedded checksum table differs from the audited official table")
        self.checksums = {}
        for line in raw.decode().splitlines():
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*").removeprefix("./")
            if relative in self.checksums:
                raise ValueError("Duplicate checksum entry")
            self.checksums[relative] = digest
        self.verified = set()

    def read(self, relative):
        member = PurePosixPath(relative)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("Unexpected relative source path")
        raw = self.z.read(self.prefix + relative)
        if hashlib.sha256(raw).hexdigest() != self.checksums.get(relative):
            raise ValueError(f"Source member hash mismatch: {relative}")
        self.verified.add(relative)
        return raw

    def close(self):
        self.z.close()


def run(archive_path, output):
    if output.exists():
        raise FileExistsError(f"Output directory exists; choose a new output path: {output}")
    archive_output = output.parent / (output.name + ".zip")
    if archive_output.exists():
        raise FileExistsError(f"Return archive already exists: {archive_output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    log = []

    def announce(message):
        log.append(message)
        print(message, flush=True)

    write_json(output / "feature_configuration.json", CONFIG)
    source = VerifiedArchive(archive_path)
    try:
        participants = list(csv.DictReader(io.StringIO(source.read("preprocessed/file_list.csv").decode())))
        ids = np.asarray([str(row["id"]).zfill(3) for row in participants])
        labels = np.asarray([int(row["label"]) for row in participants], dtype=np.int64)
        if len(ids) != 469 or len(set(ids)) != 469 or Counter(labels.tolist()) != Counter({0: 79, 1: 276, 2: 114}):
            raise ValueError("Cohort differs from the received data audit; investigate rather than silently drop rows")
        values = np.full((len(ids), 11, 2, 2, len(FEATURES)), np.nan)
        clinical = np.full((len(ids), 32), np.nan)
        planned = np.zeros((len(ids), 11))
        timestamp_cost = np.zeros_like(planned)
        quality = []
        source_rows = []
        for i, (sid, person) in enumerate(zip(ids, participants)):
            clinical[i, :2] = [float(person["age"]), {"female": 0, "male": 1}[person["gender"]]]
            questionnaire = json.loads(source.read(f"questionnaire/questionnaire_response_{sid}.json"))
            answers = {item["link_id"]: item["answer"] for item in questionnaire["item"]}
            if questionnaire["subject_id"] != sid or len(questionnaire["item"]) != 30 or sorted(answers) != [f"{j:02d}" for j in range(1, 31)] or not all(type(v) is bool for v in answers.values()):
                raise ValueError(f"Invalid questionnaire: {sid}")
            clinical[i, 2:] = [answers[f"{j:02d}"] for j in range(1, 31)]
            observation = json.loads(source.read(f"movement/observation_{sid}.json"))
            if observation["subject_id"] != sid or [s["record_name"] for s in observation["session"]] != TASKS:
                raise ValueError(f"Unexpected task identity/order: {sid}")
            for j, session in enumerate(observation["session"]):
                planned[i, j] = session["rows"] / observation["sampling_rate"]
                records = {r["device_location"]: r for r in session["records"]}
                if len(session["records"]) != 2 or set(records) != set(WRISTS):
                    raise ValueError(f"Unexpected wrist records: {sid}, {TASKS[j]}")
                durations = []
                for k, wrist in enumerate(WRISTS):
                    metadata = records[wrist]
                    channels = metadata["channels"]
                    expected = ["Time"] + [f"{sensor}_{axis}" for sensor in SENSORS for axis in "XYZ"]
                    if len(channels) != 7 or set(channels) != set(expected):
                        raise ValueError("Unexpected sensor channels")
                    units = dict(zip(channels, metadata["units"]))
                    if any(units[c] != ("s" if c == "Time" else "g" if c.startswith("Accelerometer") else "rad/s") for c in expected):
                        raise ValueError("Unexpected sensor units")
                    relative = "movement/" + metadata["file_name"]
                    raw = np.loadtxt(io.BytesIO(source.read(relative)), delimiter=",")
                    if raw.shape != (session["rows"], 7):
                        raise ValueError(f"Unexpected raw signal shape: {relative}")
                    raw = raw[:, [channels.index(c) for c in expected]]
                    t, x = raw[:, 0], raw[:, 1:]
                    descriptors, qc = extract_record(t, x)
                    values[i, j, k] = descriptors
                    dt = np.diff(t)
                    duration = float(t[-1] - t[0] + np.median(dt))
                    durations.append(duration)
                    quality.append({"subject_id": sid, "label": int(labels[i]), "task": TASKS[j], "wrist": wrist,
                                    "source_samples": len(t), "gaps_over_100ms": int((dt > 0.1).sum()),
                                    "max_gap_s": float(dt.max()), "timestamp_duration_s": duration,
                                    **{a: b for a, b in qc.items() if a != "retained_time_spans"},
                                    "retained_time_spans": json.dumps(qc["retained_time_spans"]),
                                    "descriptors_available": bool(np.isfinite(descriptors).all())})
                    source_rows.append({"subject_id": sid, "task": TASKS[j], "wrist": wrist,
                                        "relative_path": relative, "sha256": source.checksums[relative]})
                timestamp_cost[i, j] = max(durations)
            if (i + 1) % 25 == 0 or i + 1 == len(ids):
                announce(f"Extracted {i + 1}/{len(ids)} participants in {time.monotonic() - started:.1f} seconds")
        task_features = bilateral_descriptors(values)
        if not np.isfinite(clinical).all():
            raise ValueError("Invalid baseline inputs")
        output_arrays = dict(subject_ids=ids, source_labels=labels, tasks=np.asarray(TASKS), wrists=np.asarray(WRISTS),
                             sensors=np.asarray(SENSORS), descriptor_names=np.asarray(FEATURES),
                             wrist_sensor_descriptors=values, task_features=task_features,
                             task_feature_names=np.asarray(TASK_FEATURES), clinical_inputs=clinical,
                             clinical_names=np.asarray(["age", "gender_male"] + [f"nms_{j:02d}" for j in range(1, 31)]),
                             primary_mask=np.isin(labels, [1, 2]),
                             primary_y=(labels[np.isin(labels, [1, 2])] == 1).astype(np.int64),
                             planned_seconds=planned, timestamp_estimated_seconds=timestamp_cost,
                             record_available=np.isfinite(values).all(axis=(3, 4)),
                             gap_affected_records=np.asarray([q["gaps_over_100ms"] > 0 for q in quality]).reshape(len(ids), 11, 2))
        np.savez_compressed(output / "PADS_motion_features.npz", **output_arrays)
        write_csv(output / "feature_quality.csv", quality)
        write_csv(output / "motion_source_hashes.csv", source_rows)
        write_csv(output / "participant_metadata.csv", [
            {"subject_id": sid, "source_label": int(labels[i]), "condition": person["condition"],
             "age": float(clinical[i, 0]), "gender_male": int(clinical[i, 1])}
            for i, (sid, person) in enumerate(zip(ids, participants))])
        write_csv(output / "feature_dictionary.csv", [
            {"column_index_within_task": i, "name": n, "source": "current acquired task only"}
            for i, n in enumerate(TASK_FEATURES)])
        summary = {"extraction_completed": True, "model_fitting_performed": False,
                   "raw_archive_sha256": ARCHIVE_HASH, "source_manifest_sha256": MANIFEST_HASH,
                   "source_verification": "matches the previously audited ZIP and checksum table; each used source member rehashed",
                   "verified_source_members": len(source.verified), "participants": len(ids),
                   "primary_participants": int(np.isin(labels, [1, 2]).sum()),
                   "task_feature_shape": list(task_features.shape), "wrist_sensor_shape": list(values.shape),
                   "records": len(quality), "gap_affected_records": sum(q["gaps_over_100ms"] > 0 for q in quality),
                   "records_without_usable_descriptors": sum(not q["descriptors_available"] for q in quality),
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "elapsed_seconds": time.monotonic() - started,
                   "features_are_clinically_validated": False,
                   "unavailable_record_rule": "Retain participant, mark missing; choose fold-confined handling before model fitting"}
        write_json(output / "feature_summary.json", summary)
        (output / "execution.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        for name in ["extract_pads_features.py", "FEATURE_PLAN.md", "START_HERE.md", "requirements.txt",
                     "primary_outer_folds.csv", "AUDIT_ACCEPTANCE.json", "LOCAL_VERIFICATION.json"]:
            item = Path(__file__).with_name(name)
            if item.is_file():
                (output / name).write_bytes(item.read_bytes())
        members = sorted(p for p in output.iterdir() if p.is_file())
        manifest = {"bundle": "PADS_features_return", "model_fitting_performed": False,
                    "payload_files": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in members]}
        write_json(output / "RETURN_MANIFEST.json", manifest)
        with zipfile.ZipFile(archive_output, "w", zipfile.ZIP_DEFLATED) as package:
            for p in sorted(output.iterdir()):
                if p.is_file():
                    package.write(p, "PADS_features_return/" + p.name)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"RETURN FILE: {archive_output.resolve()} ({archive_output.stat().st_size} bytes)", flush=True)
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("PADS_features_return"))
    args = parser.parse_args()
    run(args.archive, args.output)


if __name__ == "__main__":
    main()
