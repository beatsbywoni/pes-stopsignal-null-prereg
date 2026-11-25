#!/usr/bin/env python3
"""Preregistered replication in the Self-Regulation Ontology retest data, stop-signal task
(registration: sro_prereg/PREREGISTRATION_SRO_stop_signal.md). Reuses the functions of stability.py and revision2.py.

Usage: python3 sro_replication.py <SRO Data folder> OUT.json [--synthetic]
  --synthetic replaces every RT by a draw from a fixed distribution (structure test only; used before registration)
"""
import importlib.util, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
os.environ.setdefault("HEDGE_RAW", os.environ.get("HEDGE_RAW", "/nonexistent"))
ST = _load("stab", "stability.py"); RV = _load("rev2", "revision2.py")
TASK = "Stopsignal"; SEED = 20261201; NBOOT = ST.NBOOT
q, r4, cor = ST.q, ST.r4, ST.cor


def load(folder, synthetic=False):
    """sess -> worker -> rows (run, trial, cond, correct, rt_s, ssd_s) in stability.py's stop-signal coding"""
    S = {}
    rng = np.random.default_rng(7)
    for s, wave in ((1, "Complete_02-16-2019"), (2, "Retest_02-16-2019")):
        df = pd.read_csv(os.path.join(folder, wave, "Individual_Measures", "stop_signal.csv.gz"), compression="gzip", low_memory=False)
        df = df[df.exp_stage == "test"]
        S[s] = {}
        for w, d in df.groupby("worker_id", sort=True):
            tn = d.trial_num.to_numpy(); run = np.concatenate([[1], 1 + np.cumsum(np.diff(tn) < 0)])
            rows = []
            for (k, r), b in zip(d.iterrows(), run):
                stop = r.SS_trial_type == "stop"; resp = r.key_press != -1
                if stop: cond, corr = "1", (0 if resp else 1)
                else: cond, corr = "0", int(resp and r.key_press == r.correct_response)
                rt = float(r.rt) / 1000.0 if (resp and r.rt > 0) else -1.0
                if synthetic and rt > 0: rt = float(rng.lognormal(np.log(0.55), 0.25))
                rows.append((int(b), int(r.trial_num), cond, corr, rt, float(r.SS_delay) / 1000.0, r.condition))
            S[s][w] = rows
    both = sorted(set(S[1]) & set(S[2]))
    return {s: {w: [x[:6] for x in S[s][w]] for w in both} for s in (1, 2)}, {s: {w: {x[0]: x[6] for x in S[s][w]} for w in both} for s in (1, 2)}


def eligible_oe(S, kind, ps):
    out = []
    for p in ps:
        ok = True
        for s in (1, 2):
            pe, pc = ST.pairs(S[s][p], TASK, kind)
            if len(pe) < ST.MIN_EV: ok = False; break
            for h in (0, 1):
                if sum(1 for k, _ in enumerate(pe) if k % 2 == h) < 2 or sum(1 for k, _ in enumerate(pc) if k % 2 == h) < 2: ok = False
            if not ok: break
        if ok: out.append(p)
    return out


def mean_ci(x, rng, level=.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bi = rng.integers(0, len(x), (NBOOT, len(x))); m = x[bi].mean(1); a = (1 - level) / 2
    return dict(n=int(len(x)), mean_ms=r4(1000 * x.mean()), CI_ms=[r4(1000 * np.quantile(m, a)), r4(1000 * np.quantile(m, 1 - a))])


def run(folder, synthetic=False):
    S, RUNCOND = load(folder, synthetic)
    ps_all = sorted(S[1])
    res = {"n_both_sessions": len(ps_all)}
    B = ST.build(TASK, S, np.random.default_rng(SEED))
    ps = eligible_oe(S, "ss_em", ps_all); res["n_primary"] = len(ps)
    idx = np.random.default_rng(SEED + 1).integers(0, len(ps), (NBOOT, len(ps)))
    out, boots = {}, {}
    out["EM"], boots["EM"] = ST.coef(B, ps, "prim", "diff", idx)
    out["POSTSIG"], boots["POSTSIG"] = ST.coef(B, ps, "postsig", "diff", idx)
    out["RT"], boots["RT"] = ST.coef(B, ps, "rt", "mean", idx)
    out["SSRT"], boots["SSRT"] = ST.coef(B, ps, "ssrt", "ssrt", idx)
    out["ER"], boots["ER"] = ST.coef(B, ps, "er", "mean", idx, scale=1.0)
    psc = [p for p in eligible_oe(S, "conv", ps_all) if p in set(ps)]
    idc = np.random.default_rng(SEED + 2).integers(0, len(psc), (NBOOT, len(psc)))
    out["CONV"], _ = ST.coef(B, psc, "conv", "diff", idc)
    out["rho_EM_minus"] = {k: dict(diff=r4(out["EM"]["rho"] - out[k]["rho"]), CI=ST.paired(boots["EM"]["rho_b"], boots[k]["rho_b"])) for k in ("POSTSIG", "RT", "SSRT", "ER")}
    # H1
    em = out["EM"]; res["H1"] = dict(rho=em["rho"], CI=em["rho_CI"], undefined=em["rho_undefined"],
                                      supported=bool(em["rho_CI"][1] < 1 and em["rho_undefined"] <= 0.05 * NBOOT))
    # local baseline (H2, H4, H5) with revision2.local_seq
    LB = {1: {}, 2: {}}; rng = np.random.default_rng(SEED + 3)
    for s in (1, 2):
        for p in ps:
            ev, ct = RV.local_seq(S[s][p], TASK, "ss_em", joint=True)
            LB[s][p] = {"lb_raw": (ev, ct), "lb": {"ev": ST.Comp([t[0] for t in ev], ["x"] * len(ev), rng), "ctl": ST.Comp([t[0] for t in ct], ["x"] * len(ct), rng)},
                        "prim": B[s][p]["prim"]}
    psl = [p for p in ps if all(len(LB[s][p]["lb_raw"][0]) >= ST.MIN_EV and len(LB[s][p]["lb_raw"][1]) >= ST.MIN_EV for s in (1, 2))]
    idl = np.random.default_rng(SEED + 4).integers(0, len(psl), (NBOOT, len(psl)))
    out["LB"], bl = ST.coef(LB, psl, "lb", "diff", idl)
    out["EM_same_as_LB"], br = ST.coef(LB, psl, "prim", "diff", idl)
    h5 = {}
    for s, k in ((1, "rb1"), (2, "rb2")):
        d = bl[k] - br[k]; h5[f"s{s}"] = dict(rel_local=out["LB"][f"rel{s}"], rel_raw=out["EM_same_as_LB"][f"rel{s}"], diff_CI=q(d))
    res["H5"] = dict(n=len(psl), sessions=h5, supported=bool(all(h5[f"s{s}"]["diff_CI"][1] < 0 for s in (1, 2))))
    means, h2, h3, h4 = {}, {}, {}, {}
    for s in (1, 2):
        f = {k: [] for k in ("local", "e1", "pre1", "pre4", "post")}
        for p in ps:
            ev, ct = LB[s][p]["lb_raw"]
            if len(ev) < 2 or len(ct) < 2:
                for k in f: f[k].append(np.nan)
                continue
            m = lambda L, j: np.mean([t[j] for t in L])
            f["local"].append(m(ev, 0) - m(ct, 0)); f["e1"].append((m(ev, 6) - m(ev, 4)) - (m(ct, 6) - m(ct, 4)))
            f["pre1"].append(m(ev, 4) - m(ct, 4)); f["pre4"].append(m(ev, 5) - m(ct, 5)); f["post"].append(m(ev, 6) - m(ct, 6))
        r_ = np.random.default_rng(SEED + 10 + s)
        means[f"s{s}"] = {k: mean_ci(v, r_) for k, v in f.items()}
        loc90 = mean_ci(f["local"], np.random.default_rng(SEED + 20 + s), level=.90)
        h2[f"s{s}"] = dict(mean_ms=means[f"s{s}"]["local"]["mean_ms"], CI95=means[f"s{s}"]["local"]["CI_ms"], CI90=loc90["CI_ms"])
        h4[f"s{s}"] = means[f"s{s}"]["pre1"]
        # post-signal slowing raw and local baseline
        raw = np.array([ST.score_full("diff", B[s][p]["postsig"]) for p in ps])
        lbp = []
        for p in ps:
            ev, ct = RV.local_seq(S[s][p], TASK, "postsig_ss", joint=False)
            lbp.append(np.mean([t[0] for t in ev]) - np.mean([t[0] for t in ct]) if len(ev) >= ST.MIN_EV and len(ct) >= ST.MIN_EV else np.nan)
        r2 = np.random.default_rng(SEED + 30 + s)
        h3[f"s{s}"] = dict(raw=mean_ci(raw, r2), local=mean_ci(lbp, r2))
        means[f"s{s}"]["EM_raw"] = mean_ci([ST.score_full("diff", B[s][p]["prim"]) for p in ps], r2)
        means[f"s{s}"]["CONV"] = mean_ci([ST.score_full("diff", B[s][p]["conv"]) for p in psc], r2)
    res["H2"] = dict(sessions=h2, supported=bool(all(-15 < h2[f"s{s}"]["CI90"][0] and h2[f"s{s}"]["CI90"][1] < 15 for s in (1, 2))),
                     contradicted=bool(any(h2[f"s{s}"]["CI95"][0] > 0 for s in (1, 2))))
    res["H3"] = dict(sessions=h3, supported=bool(all(h3[f"s{s}"][k]["CI_ms"][0] > 0 for s in (1, 2) for k in ("raw", "local"))))
    res["H4"] = dict(sessions=h4, supported=bool(all(h4[f"s{s}"]["CI_ms"][1] < 0 for s in (1, 2))))
    # H6: raw post-signal slowing by run (high = 40% stop, low = 20% stop); further: local baseline by run
    h6 = {}
    for s in (1, 2):
        d, dl = [], []
        for p in ps:
            rows = S[s][p]; rc = RUNCOND[s][p]; v = {}
            for c in ("high", "low"):
                sub = [r for r in rows if rc[r[0]] == c]
                se, sc = ST.pairs(sub, TASK, "postsig_ss")
                v[c] = (np.mean([t[0] for t in se]) - np.mean([t[0] for t in sc])) if len(se) >= 2 and len(sc) >= 2 else np.nan
                ev, ct = RV.local_seq(sub, TASK, "postsig_ss", joint=False)
                v[c + "_lb"] = (np.mean([t[0] for t in ev]) - np.mean([t[0] for t in ct])) if len(ev) >= 2 and len(ct) >= 2 else np.nan
            d.append(v["high"] - v["low"]); dl.append(v["high_lb"] - v["low_lb"])
        r6 = np.random.default_rng(SEED + 40 + s)
        h6[f"s{s}"] = dict(raw_high_minus_low=mean_ci(d, r6), local_high_minus_low=mean_ci(dl, r6))
    pos = all(h6[f"s{s}"]["raw_high_minus_low"]["CI_ms"][0] > 0 for s in (1, 2)); neg = all(h6[f"s{s}"]["raw_high_minus_low"]["CI_ms"][1] < 0 for s in (1, 2))
    res["H6"] = dict(sessions=h6, verdict="goal priority" if pos else ("surprise" if neg else "inconclusive"))
    res["coefficients"] = out; res["means"] = means
    res["precision_weighted_EM"] = RV.a4(TASK, S, ps, 0)
    res["lag_profile"] = RV.a6(TASK, S, ps, 0)
    keep = [p for p in ps if RV.consensus_ok(S[1][p]) and RV.consensus_ok(S[2][p])]
    ik = np.random.default_rng(SEED + 5).integers(0, len(keep), (NBOOT, len(keep)))
    res["sensitivity_consensus"] = dict(n=len(keep)); res["sensitivity_consensus"]["EM"], _ = ST.coef(B, keep, "prim", "diff", ik)
    res["change"] = {k: dict(change=out[k]["change"], CI=out[k]["change_CI"]) for k in ("EM", "POSTSIG", "RT", "SSRT")}
    res["persons"] = ps
    return res


if __name__ == "__main__":
    folder, OUT = sys.argv[1], sys.argv[2]; syn = "--synthetic" in sys.argv
    res = run(folder, syn); res["synthetic"] = syn
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_both_sessions", "n_primary", "synthetic")}))
