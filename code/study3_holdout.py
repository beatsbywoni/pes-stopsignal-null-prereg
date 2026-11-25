#!/usr/bin/env python3
"""Study 3 (preregistered): does slowing after failed stops exceed what the speed around stop trials predicts?
Holdout of the Self-Regulation Ontology stop-signal data: participants of the first wave who did not take part in the retest.
The code reuses the trial coding of sro_replication.py and the definitions of Addendum C2 (C5, C6) of the earlier analyses.

Usage: python3 study3_holdout.py <SRO Data folder> OUT.json [--synthetic]
With --synthetic every RT is replaced by a random draw (used to test the code before registration)."""
import importlib.util, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
SR = _load("sro", "sro_replication.py"); ST = SR.ST; RV = SR.RV
TASK = "Stopsignal"; LAGS = [-4, -3, -2, -1, 1, 2, 3, 4, 5]; NB = 2000; SEED = 20261204
FE, FC = ST.PRED["ss_em"]
r4 = ST.r4


def load_holdout(folder, synthetic=False):
    """worker -> rows (run, trial, cond, correct, rt_s, ssd_s), same coding as sro_replication.load; first wave only,
    participants without test-stage data in the retest wave"""
    rng = np.random.default_rng(7)
    a = pd.read_csv(os.path.join(folder, "Complete_02-16-2019", "Individual_Measures", "stop_signal.csv.gz"), compression="gzip", low_memory=False)
    b = pd.read_csv(os.path.join(folder, "Retest_02-16-2019", "Individual_Measures", "stop_signal.csv.gz"), compression="gzip", usecols=["worker_id", "exp_stage"], low_memory=False)
    retest = set(b[b.exp_stage == "test"].worker_id)
    a = a[(a.exp_stage == "test") & ~a.worker_id.isin(retest)]
    S = {}
    for w, d in a.groupby("worker_id", sort=True):
        tn = d.trial_num.to_numpy(); run = np.concatenate([[1], 1 + np.cumsum(np.diff(tn) < 0)])
        rows = []
        for (k, r), bb in zip(d.iterrows(), run):
            stop = r.SS_trial_type == "stop"; resp = r.key_press != -1
            if stop: cond, corr = "1", (0 if resp else 1)
            else: cond, corr = "0", int(resp and r.key_press == r.correct_response)
            rt = float(r.rt) / 1000.0 if (resp and r.rt > 0) else -1.0
            if synthetic and rt > 0: rt = float(rng.lognormal(np.log(0.55), 0.25))
            rows.append((int(bb), int(r.trial_num), cond, corr, rt, float(r.SS_delay) / 1000.0))
        S[w] = rows
    return S


def eligible(S):
    """at least 10 scored go trials after failed stops, and two after failed and two after successful stops in both
    odd/even halves (the rule of sro_replication.eligible_oe, applied to one session)"""
    out = []
    for p, rows in S.items():
        pe, pc = ST.pairs(rows, TASK, "ss_em")
        if len(pe) < ST.MIN_EV: continue
        if all(sum(1 for k, _ in enumerate(x) if k % 2 == h) >= 2 for x in (pe, pc) for h in (0, 1)): out.append(p)
    return out


def ssrt(rows):
    """integration SSRT, go omissions replaced by the slowest go RT, runs pooled (as in Study 2)"""
    go = [r for r in rows if r[2] == "0"]; stp = [r for r in rows if r[2] == "1"]
    g = np.array([r[4] if r[4] > 0.05 else np.nan for r in go]); g[np.isnan(g)] = np.nanmax(g); g = np.sort(g)
    p = np.mean([1 - r[3] for r in stp]); k = int(np.clip(math.ceil(p * len(g)) - 1, 0, len(g) - 1))
    return g[k] - np.mean([r[5] for r in stp])


def prof(rows, events):
    acc = {"e": {k: [] for k in LAGS}, "c": {k: [] for k in LAGS}}
    for n, t in events:
        for k in LAGS:
            if RV.ok_go(rows, n + k, rows[n][0], TASK): acc[t][k].append(rows[n + k][4])
    return [(np.mean(acc["e"][k]) - np.mean(acc["c"][k])) if len(acc["e"][k]) >= 2 and len(acc["c"][k]) >= 2 else np.nan for k in LAGS]


def person_profiles(rows, shift=0.0):
    """observed (failed minus successful stop) and null (pseudo-failure minus pseudo-success) profiles, in s"""
    t_ssrt = ssrt(rows) + shift
    ev = [(n, "e" if FE(TASK, r) else "c") for n, r in enumerate(rows) if FE(TASK, r) or FC(TASK, r)]
    pse = []
    for n, r in enumerate(rows):
        if not RV.ok_go(rows, n, r[0], TASK): continue
        nxt = next((rows[m][5] for m in range(n + 1, len(rows)) if rows[m][0] == r[0] and rows[m][2] == "1"), None)
        if nxt is None: continue
        pse.append((n, "e" if r[4] < nxt + t_ssrt else "c"))
    return prof(rows, ev), prof(rows, pse), float(np.mean([t == "e" for _, t in pse]))


def analyze(S, ps, rng, shift=0.0):
    O, N0, share = [], [], []
    for p in ps:
        o, z, sh = person_profiles(S[p], shift); O.append(o); N0.append(z); share.append(sh)
    O, N0 = np.array(O) * 1000, np.array(N0) * 1000
    ok = ~np.isnan(O[:, :5]).any(1) & ~np.isnan(N0[:, :5]).any(1)
    O, N0 = O[ok], N0[ok]; n = len(O); bi = rng.integers(0, n, (NB, n))
    i1, im1, im4 = LAGS.index(1), LAGS.index(-1), LAGS.index(-4); late = [LAGS.index(k) for k in (2, 3, 4, 5)]
    def stat(o, z):
        mo, mz = np.nanmean(o, 0), np.nanmean(z, 0); d = mo - mz
        return np.concatenate([mo, mz, d, [d[i1], d[i1] - d[im1], (mo[i1] - mo[im4]) - (mz[i1] - mz[im4]), mz[i1] - mz[im4], mo[i1] - mo[im4], np.mean(d[late])]])
    pt = stat(O, N0); B = np.array([stat(O[b], N0[b]) for b in bi]); lo, hi = np.nanquantile(B, .025, 0), np.nanquantile(B, .975, 0)
    L = len(LAGS); cell = lambda i: dict(mean=r4(pt[i]), CI=[r4(lo[i]), r4(hi[i])])
    return dict(n=n, pseudo_failure_share=r4(float(np.mean(np.array(share)[ok]))),
                observed={str(k): cell(i) for i, k in enumerate(LAGS)}, null={str(k): cell(L + i) for i, k in enumerate(LAGS)},
                observed_minus_null={str(k): cell(2 * L + i) for i, k in enumerate(LAGS)},
                H7a_plus1=cell(3 * L), H7b_plus1_minus_minus1=cell(3 * L + 1), H8_local_observed_minus_null=cell(3 * L + 2),
                local_null=cell(3 * L + 3), local_observed=cell(3 * L + 4), H9_late_mean=cell(3 * L + 5))


def windows_local(S, ps, rng):
    """registered local-baseline score of Study 2 (non-overlapping windows, RT(n+1) - RT(n-4))"""
    loc = []
    for p in ps:
        ev, ct = RV.local_seq(S[p], TASK, "ss_em", joint=True)
        loc.append(np.mean([t[0] for t in ev]) - np.mean([t[0] for t in ct]) if len(ev) >= 2 and len(ct) >= 2 else np.nan)
    return dict(local=SR.mean_ci(loc, rng), local90=SR.mean_ci(loc, rng, .90))


def run(folder, synthetic=False):
    S = load_holdout(folder, synthetic); ps_all = sorted(S); ps = eligible(S)
    res = dict(synthetic=synthetic, n_holdout=len(ps_all), n_eligible=len(ps))
    res["primary"] = analyze(S, ps, np.random.default_rng(SEED))
    P = res["primary"]
    res["H7_supported"] = bool(P["H7a_plus1"]["CI"][0] > 0 and P["H7b_plus1_minus_minus1"]["CI"][0] > 0)
    res["H7_contradicted"] = bool(P["H7a_plus1"]["CI"][1] < 0)
    res["H8_supported"] = bool(P["H8_local_observed_minus_null"]["CI"][0] > 0)
    res["H9_supported"] = bool(P["H9_late_mean"]["CI"][1] < 0)
    res["H10_selection_check"] = bool(P["observed"]["-1"]["CI"][1] < 0)
    res["calibration"] = {k: dict(obs_minus_null=P["observed_minus_null"][k], includes_zero=bool(P["observed_minus_null"][k]["CI"][0] <= 0 <= P["observed_minus_null"][k]["CI"][1])) for k in ("-4", "-3", "-2", "-1")}
    # sensitivity: consensus criteria (Verbruggen et al., 2019) and SSRT shifted by -25 and +25 ms in the null
    keep = [p for p in ps if RV.consensus_ok(S[p])]
    res["sensitivity_consensus"] = dict(n_requested=len(keep), result=analyze(S, keep, np.random.default_rng(SEED + 1)))
    res["sensitivity_ssrt_minus25"] = analyze(S, ps, np.random.default_rng(SEED + 2), shift=-0.025)
    res["sensitivity_ssrt_plus25"] = analyze(S, ps, np.random.default_rng(SEED + 3), shift=+0.025)
    # descriptive: registered local-baseline score of Study 2 and raw event-matched PES
    rng = np.random.default_rng(SEED + 4)
    res["descriptive_windows"] = windows_local(S, ps, rng)
    em = [np.mean([t[0] for t in ST.pairs(S[p], TASK, "ss_em")[0]]) - np.mean([t[0] for t in ST.pairs(S[p], TASK, "ss_em")[1]]) for p in ps]
    res["descriptive_raw_EM"] = SR.mean_ci(em, rng)
    res["persons"] = ps
    return res


if __name__ == "__main__":
    folder, outp = sys.argv[1], sys.argv[2]; syn = "--synthetic" in sys.argv
    r = run(folder, syn)
    json.dump(r, open(outp, "w"), indent=1)
    print("STUDY3", "synthetic" if syn else "REAL", json.dumps({k: v for k, v in r.items() if k.startswith("H") or k.startswith("n_")}))
