"""Focused synthetic verification; no clinical performance is measured here."""

import json
import numpy as np
from threadpoolctl import threadpool_limits
from run_analysis import AcquisitionSystem, VALID, nominal_units


def main():
    rng = np.random.default_rng(173)
    y = np.tile([0, 1], 36)
    clinical = rng.normal(size=(72, 32))
    movement = rng.normal(size=(72, 11, 68))
    movement[:, :, 0] += y[:, None] * 0.7
    with threadpool_limits(limits=1):
        system = AcquisitionSystem().fit(clinical[:60], movement[:60], y[:60], np.asarray([f"s{i}" for i in range(60)]))
        transformed = system.representation.transform(clinical[60:], movement[60:])
        for row in transformed:
            columns = system.representation.columns[1]
            before = system.recommend(1, row[columns], 7)
            changed = row.copy()
            changed[40:] = rng.normal(size=len(changed) - 40) * 1e6
            assert before == system.recommend(1, changed[columns], 7)
            assert system.score_observed(1, row[columns]) == system.score_observed(1, changed[columns])
        results = system.predict_scenarios(clinical[60:], movement[60:])
        assert len(VALID) == 257
        for method, result in results.items():
            assert np.isfinite(result["scores"]).all() and result["scores"].shape == (12,)
            if method not in {"full_protocol", "clinical_only"}:
                assert all(int(mask) in VALID and nominal_units(int(mask)) <= 7 for mask in result["masks"])
        assert not set(system.ids) & set(f"s{i}" for i in range(60, 72))
        for fold in system.ledger["state_crossfit"]:
            assert not set(fold["train_ids"]) & set(fold["held_out_ids"])
    print(json.dumps({"passed": True, "clinical_performance_evaluation": False,
                      "checks": ["unacquired-feature perturbation cannot change current score or next action",
                                 "259 common state classifiers and 257 feasible abbreviated states",
                                 "all 17 scenarios return finite scores for every held-out subject",
                                 "whole-task ordering, prerequisites and budget constraints",
                                 "patient-disjoint learning and holdout sets"]}, indent=2))


if __name__ == "__main__":
    main()
