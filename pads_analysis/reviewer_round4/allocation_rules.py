"""Two-cutoff rejection allocation; inference never receives reference labels."""
import numpy as np

def apply_allocation(binary, calibrated_probability, cutoff_predicted_dd, cutoff_predicted_pd):
    binary = np.asarray(binary, dtype=int)
    p = np.asarray(calibrated_probability, dtype=float)
    if p.shape != binary.shape or not np.isfinite(p).all():
        raise ValueError("Matching finite vectors are required")
    if not np.isin(binary, [0,1]).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid prediction")
    confidence = np.where(binary==1,p,1-p)
    cutoff = np.where(binary==1,cutoff_predicted_pd,cutoff_predicted_dd)
    return np.where(confidence<cutoff,-1,binary)

def candidate_allocations(binary, calibrated_probability, labels, target, cost):
    """Enumerate every achievable allocation at the largest permitted count."""
    binary, p, y = np.asarray(binary), np.asarray(calibrated_probability), np.asarray(labels)
    n = len(y)
    assert p.shape == binary.shape == y.shape and set(np.unique(y))=={0,1}
    kmax = int(np.floor(target*n))
    nc = np.bincount(y,minlength=2)
    wrong = np.array([np.sum((binary!=y)&(y==c)) for c in [0,1]])
    confidence = np.where(binary==1,p,1-p)
    states = []
    for predicted in [0,1]:
        ix = np.flatnonzero(binary==predicted)
        ix = ix[np.argsort(confidence[ix],kind="stable")]
        cs = confidence[ix]
        if len(ix):
            _, count = np.unique(cs,return_counts=True)
            possible = np.r_[0,np.cumsum(count)]
        else:
            possible = np.array([0])
        prefixes = np.stack([np.r_[0,np.cumsum(y[ix]==c)] for c in [0,1]],axis=1)
        rows = []
        for k in possible[possible<=kmax]:
            k = int(k)
            cutoff = 0. if k==0 else (float(np.nextafter(1.,2.)) if k==len(ix) else float(cs[k]))
            deferred = prefixes[k]
            removed = deferred*np.array([predicted!=c for c in [0,1]])
            rows.append((k,cutoff,deferred,removed))
        states.append(rows)
    totals = [a[0]+b[0] for a in states[0] for b in states[1] if a[0]+b[0]<=kmax]
    total = max(totals)
    candidates = []
    for a in states[0]:
        for b in states[1]:
            if a[0]+b[0]!=total:
                continue
            deferred = a[2]+b[2]
            errors = wrong-a[3]-b[3]
            loss = (errors+cost*deferred)/nc
            candidates.append({
                "k_predicted_dd":a[0],"k_predicted_pd":b[0],
                "cutoff_predicted_dd":a[1],"cutoff_predicted_pd":b[1],
                "development_n":n,"development_pd":int(nc[1]),"development_dd":int(nc[0]),
                "development_rejected":total,"nominal_rejected_count":kmax,
                "development_pd_clear_wrong":int(errors[1]),"development_dd_clear_wrong":int(errors[0]),
                "development_pd_deferred":int(deferred[1]),"development_dd_deferred":int(deferred[0]),
                "development_pd_loss":float(loss[1]),"development_dd_loss":float(loss[0]),
                "development_mean_loss":float(loss.mean()),"development_worst_loss":float(loss.max()),
                "development_deferral_gap":float(abs(deferred[1]/nc[1]-deferred[0]/nc[0])),
            })
    assert candidates
    return candidates

def choose_allocation(candidates,objective):
    if objective=="mean_class_loss":
        criteria=["development_mean_loss","development_worst_loss"]
    elif objective=="worst_class_loss":
        criteria=["development_worst_loss","development_mean_loss"]
    else:
        raise ValueError("Unknown allocation objective")
    eligible = list(candidates)
    for key in criteria+["development_deferral_gap"]:
        best = min(row[key] for row in eligible)
        eligible = [row for row in eligible if row[key]<=best+1e-12]
    selected = min(eligible,key=lambda row:row["k_predicted_dd"])
    return dict(selected)
