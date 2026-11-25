#!/usr/bin/env python3
"""Fourth-round analyses (plan: plans/ANALYSIS_PLAN_revision.md, Addendum A; exploratory).

A1. Local-baseline event-matched scores in stop-signal and go/no-go: for each stop or no-go trial n whose trials
    n - 4, n - 1, and n + 1 are correct go trials with valid RT in the same block (windows n - 4 .. n + 1 that share a
    trial with an earlier retained window are dropped, over failed and successful events together), the value
    RT(n + 1) - RT(n - 4); score = mean after failed minus mean after successful inhibition. Also reported: the
    pre-event differences RT(n - 1) and RT(n - 4) (failed minus successful), and RT(n + 1) - RT(n - 1).
    Post-signal slowing with the same local baseline (after successful stop or correct rejection minus after a correct
    go trial; non-overlap applied within each set) is reported as means only.
A2. Local error expectancy with matched control trials: within each person and session, post-event and control
    trials are split by the number of errors in the 20 trials before the preceding trial (threshold: the person's
    median over post-event trials); PES in the locally rare group minus PES in the locally frequent group.
A3. Mean disattenuated correlation over Stroop and flanker only (conventional PES and mean correct RT), joint
    bootstrap over unique participants, as in stability.py.
A3. (see A8)
A4. Latent between-session correlation of the primary PES score from a bivariate normal model of person-session
    scores with known sampling variances (y_ps = theta_ps + e_ps, var(e_ps) = s_ev^2 / n_ev + s_ctl^2 / n_ctl;
    (theta_p1, theta_p2) ~ N(mu, Sigma)); maximum likelihood; 95% intervals from 2,000 person bootstrap resamples.

A5. Time-separated coefficients: PES and mean correct RT from blocks 1-2 and blocks 4-5 of each session (persons
    with at least 6 post-event and 6 control trials in each of the four subsets); split-half coefficients within each
    subset (stratified random splits), correlations and disattenuated correlations for S1 early-late, S2 early-late,
    S1 early-S2 early, S1 late-S2 early, S1 late-S2 late, and paired differences (within minus between).
A6. RT at lags -4 to +3 around the event and control trials (correct go or response-requiring trials with valid RT in
    the same block), event minus control per lag.
A7. Error rate with PES across persons: log-scored PES, and PES and error rate each residualized on mean correct RT.
A8. Classical estimator, sensitivity: leave-one-out range of rho; winsorized (3.5 robust SD, half scores of every
    split and full scores) and rank-based versions; block-split coefficients in both sessions and the corresponding
    rho; stop-signal persons passing consensus checks in both sessions; and (A3) the mean rho over Stroop and flanker.

Usage: HEDGE_RAW=<.../RawData> python3 revision2.py results/stability.json OUT.json
"""
import importlib.util, json, os, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("stab", os.path.join(HERE, "stability.py"))
ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
LO, HI, NBOOT, SEED = ST.LO, ST.HI, ST.NBOOT, 20261127
q, r4, cor = ST.q, ST.r4, ST.cor


# ---------------------------------------------------------------- A1
def ok_go(rows, j, b, task):
    return 0 <= j < len(rows) and rows[j][0] == b and rows[j][3] == 1 and ST.needs_resp(task, rows[j][2]) and LO <= rows[j][4] <= HI


def local_seq(rows, task, kind, joint=True):
    """event and control sequences: (RT(n+1)-RT(n-4), 'x', block, n, RT(n-1), RT(n-4), RT(n+1))"""
    fe, fc = ST.PRED[kind]
    ev, ct, last = [], [], {"e": -10, "c": -10, "j": -10}
    for n in range(4, len(rows) - 1):
        p = rows[n]; b = p[0]
        is_e, is_c = fe(task, p), fc(task, p)
        if not (is_e or is_c): continue
        if not (ok_go(rows, n - 4, b, task) and ok_go(rows, n - 1, b, task) and ok_go(rows, n + 1, b, task)): continue
        key = "j" if joint else ("e" if is_e else "c")
        if n - 4 <= last[key]: continue
        last[key] = n + 1
        item = (rows[n + 1][4] - rows[n - 4][4], "x", b, n, rows[n - 1][4], rows[n - 4][4], rows[n + 1][4])
        (ev if is_e else ct).append(item)
    return ev, ct


def boot_mean(x, rng):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bi = rng.integers(0, len(x), (NBOOT, len(x)))
    return dict(n=int(len(x)), mean_ms=r4(1000 * x.mean()), CI_ms=[r4(1000 * v) for v in q(x[bi].mean(1))])


def a1(task, S, ps, ti):
    kind = ST.PRIMARY[task]; post = "postsig_ss" if task == "Stopsignal" else "postsig_gng"
    rng = np.random.default_rng(SEED + 10 * ti)
    B = {1: {}, 2: {}}
    for s in (1, 2):
        for p in ps:
            ev, ct = local_seq(S[s][p], task, kind, joint=True)
            B[s][p] = {"em4_raw": (ev, ct),
                       "em4": {"ev": ST.Comp([t[0] for t in ev], ["x"] * len(ev), rng), "ctl": ST.Comp([t[0] for t in ct], ["x"] * len(ct), rng)}}
            pe, pc = ST.pairs(S[s][p], task, kind)
            B[s][p]["prim"] = {"ev": ST.Comp([t[0] for t in pe], ["x"] * len(pe), rng), "ctl": ST.Comp([t[0] for t in pc], ["x"] * len(pc), rng)}
    pse = ST.eligible(task, B, "em4_raw")
    idx = np.random.default_rng(SEED + 10 * ti + 1).integers(0, len(pse), (NBOOT, len(pse)))
    out = {"n_primary": len(ps), "n_local": len(pse)}
    out["EM_local"], bl = ST.coef(B, pse, "em4", "diff", idx)
    out["EM_raw_same_persons"], br = ST.coef(B, pse, "prim", "diff", idx)
    out["rho_local_minus_raw"] = dict(diff=r4(out["EM_local"]["rho"] - out["EM_raw_same_persons"]["rho"]), CI=ST.paired(bl["rho_b"], br["rho_b"]))
    out["sequences_median"] = {f"s{s}": [float(np.median([len(B[s][p]["em4_raw"][k]) for p in pse])) for k in (0, 1)] for s in (1, 2)}
    desc = {}
    for s in (1, 2):
        f = {k: [] for k in ("local", "e1", "pre1", "pre4", "post")}
        for p in pse:
            ev, ct = B[s][p]["em4_raw"]
            m = lambda L, j: np.mean([t[j] for t in L])
            f["local"].append(m(ev, 0) - m(ct, 0))
            f["e1"].append((m(ev, 6) - m(ev, 4)) - (m(ct, 6) - m(ct, 4)))
            f["pre1"].append(m(ev, 4) - m(ct, 4)); f["pre4"].append(m(ev, 5) - m(ct, 5)); f["post"].append(m(ev, 6) - m(ct, 6))
        r = np.random.default_rng(SEED + 10 * ti + 2 + s)
        desc[f"s{s}"] = {k: boot_mean(v, r) for k, v in f.items()}
        # post-signal slowing with the same local baseline (means only)
        ps_ = []
        for p in ps:
            ev, ct = local_seq(S[s][p], task, post, joint=False)
            ps_.append(np.mean([t[0] for t in ev]) - np.mean([t[0] for t in ct]) if len(ev) >= ST.MIN_EV and len(ct) >= ST.MIN_EV else np.nan)
        desc[f"s{s}"]["postsig_local"] = boot_mean(ps_, r)
    out["means"] = desc
    # deviation (added after the smoke run showed few eligible go/no-go persons): means over all primary-sample
    # persons with at least 2 windows of each type in the session
    da = {}
    for s in (1, 2):
        f = {k: [] for k in ("local", "e1", "pre1", "pre4", "post")}
        for p in ps:
            ev, ct = local_seq(S[s][p], task, kind, joint=True)
            if len(ev) < 2 or len(ct) < 2: continue
            m = lambda L, j: np.mean([t[j] for t in L])
            f["local"].append(m(ev, 0) - m(ct, 0)); f["e1"].append((m(ev, 6) - m(ev, 4)) - (m(ct, 6) - m(ct, 4)))
            f["pre1"].append(m(ev, 4) - m(ct, 4)); f["pre4"].append(m(ev, 5) - m(ct, 5)); f["post"].append(m(ev, 6) - m(ct, 6))
        r = np.random.default_rng(SEED + 10 * ti + 12 + s)
        da[f"s{s}"] = {k: boot_mean(v, r) for k, v in f.items()}
    out["means_all_primary"] = da
    out["persons"] = pse
    return out


# ---------------------------------------------------------------- A2
def a2(task, S, ps, ti):
    kind = ST.PRIMARY[task]; res = {}
    for s in (1, 2):
        d = []
        for p in ps:
            rows = S[s][p]; pe, pc = ST.pairs(rows, task, kind)
            cnt = lambda t: sum(1 for r in rows[max(0, t[3] - 20):t[3]] if r[3] == 0)
            ce = np.array([cnt(t) for t in pe]); cc = np.array([cnt(t) for t in pc])
            ve = np.array([t[0] for t in pe]); vc = np.array([t[0] for t in pc])
            if len(ve) == 0: d.append(np.nan); continue
            med = np.median(ce)
            cells = [ve[ce <= med], vc[cc <= med], ve[ce > med], vc[cc > med]]
            if min(len(c) for c in cells) < 3: d.append(np.nan); continue
            d.append((cells[0].mean() - cells[1].mean()) - (cells[2].mean() - cells[3].mean()))
        res[f"s{s}"] = boot_mean(d, np.random.default_rng(SEED + 10 * ti + 6 + s))
    return res


# ---------------------------------------------------------------- A4
def nll(par, y, v):
    mu = par[:2]; l11, l21, l22 = np.exp(par[2]), par[3], np.exp(par[4])
    L = np.array([[l11, 0.0], [l21, l22]]); Sig = L @ L.T
    tot = 0.0
    for yi, vi in zip(y, v):
        Vi = Sig + np.diag(vi); r = yi - mu
        sgn, ld = np.linalg.slogdet(Vi)
        tot += 0.5 * (ld + r @ np.linalg.solve(Vi, r))
    return tot


def nll_vec(par, y, v):
    mu = par[:2]; l11, l21, l22 = np.exp(par[2]), par[3], np.exp(par[4])
    s11, s12, s22 = l11 ** 2, l11 * l21, l21 ** 2 + l22 ** 2
    a = s11 + v[:, 0]; c = s22 + v[:, 1]; det = a * c - s12 ** 2
    r1 = y[:, 0] - mu[0]; r2 = y[:, 1] - mu[1]
    quad = (c * r1 ** 2 - 2 * s12 * r1 * r2 + a * r2 ** 2) / det
    return 0.5 * np.sum(np.log(det) + quad)


def fit_latent(y, v, start=None):
    if start is None:
        cv = np.cov(y.T); s1, s2 = np.sqrt(max(cv[0, 0] - v[:, 0].mean(), 1e-8)), np.sqrt(max(cv[1, 1] - v[:, 1].mean(), 1e-8))
        rr = np.clip(cv[0, 1] / (s1 * s2), -.95, .95)
        start = np.array([y[:, 0].mean(), y[:, 1].mean(), np.log(s1), rr * s2, np.log(s2 * np.sqrt(1 - rr ** 2))])
    best = None
    for st in (start, start + np.array([0, 0, 0, -0.5 * np.exp(start[4]), 0]), start + np.array([0, 0, 0, 0.5 * np.exp(start[4]), 0])):
        r = minimize(nll_vec, st, args=(y, v), method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12))
        r = minimize(nll_vec, r.x, args=(y, v), method="BFGS")
        if best is None or r.fun < best.fun: best = r
    l11, l21, l22 = np.exp(best.x[2]), best.x[3], np.exp(best.x[4])
    s11, s12, s22 = l11 ** 2, l11 * l21, l21 ** 2 + l22 ** 2
    return s12 / np.sqrt(s11 * s22), dict(var1=s11, var2=s22, cov=s12, mu=best.x[:2]), best.x


def a4(task, S, ps, ti):
    kind = ST.PRIMARY[task]
    y = np.zeros((len(ps), 2)); v = np.zeros((len(ps), 2))
    for j, p in enumerate(ps):
        for s in (1, 2):
            pe, pc = ST.pairs(S[s][p], task, kind)
            e = np.array([t[0] for t in pe]); c = np.array([t[0] for t in pc])
            y[j, s - 1] = 1000 * (e.mean() - c.mean())
            v[j, s - 1] = 1e6 * (e.var(ddof=1) / len(e) + c.var(ddof=1) / len(c))
    rho, par, x0 = fit_latent(y, v)
    idx = np.random.default_rng(SEED + 10 * ti + 9).integers(0, len(ps), (NBOOT, len(ps)))
    rb = np.array([fit_latent(y[i], v[i], start=x0)[0] for i in idx])
    rel = [par["var1"] / (par["var1"] + v[:, 0].mean()), par["var2"] / (par["var2"] + v[:, 1].mean())]
    return dict(n=len(ps), rho_latent=r4(rho), CI=q(rb), sd_true_ms=[r4(np.sqrt(par["var1"])), r4(np.sqrt(par["var2"]))],
                mean_sampling_sd_ms=[r4(np.sqrt(v[:, 0].mean())), r4(np.sqrt(v[:, 1].mean()))],
                reliability_implied=[r4(x) for x in rel], r_observed=r4(cor(y[:, 0], y[:, 1])))


# ---------------------------------------------------------------- A5
MIN_SUB = 6


def a5(task, S, ps, ti):
    """time-separated coefficients: blocks 1-2 versus blocks 4-5 within and between sessions (primary PES, mean RT)"""
    kind = ST.PRIMARY[task]; rng = np.random.default_rng(SEED + 10 * ti + 20)
    SUB = {"E": (1, 2), "L": (4, 5)}
    C = {}
    for s in (1, 2):
        for p in ps:
            rows = S[s][p]; pe, pc = ST.pairs(rows, task, kind)
            corr = [(r[4], ST.STRAT(task, r[2]), r[0]) for r in rows if r[3] == 1 and ST.needs_resp(task, r[2]) and LO <= r[4] <= HI]
            for lab, blks in SUB.items():
                e = [t for t in pe if t[2] in blks]; c = [t for t in pc if t[2] in blks]; x = [t for t in corr if t[2] in blks]
                C[(s, lab, p)] = dict(n_ev=len(e), n_ctl=len(c),
                                      pes={"ev": ST.Comp([t[0] for t in e], [t[1] for t in e], rng), "ctl": ST.Comp([t[0] for t in c], [t[1] for t in c], rng)},
                                      rt={"x": ST.Comp([t[0] for t in x], [t[1] for t in x], rng)})
    pse = [p for p in ps if all(C[(s, lab, p)]["n_ev"] >= MIN_SUB and C[(s, lab, p)]["n_ctl"] >= MIN_SUB for s in (1, 2) for lab in SUB)]
    idx = np.random.default_rng(SEED + 10 * ti + 21).integers(0, len(pse), (NBOOT, len(pse)))
    pairs_ = {"S1E_S1L": ((1, "E"), (1, "L")), "S2E_S2L": ((2, "E"), (2, "L")), "S1E_S2E": ((1, "E"), (2, "E")),
              "S1L_S2E": ((1, "L"), (2, "E")), "S1L_S2L": ((1, "L"), (2, "L"))}
    out = {"n": len(pse), "persons": pse}
    for meas, kd, sc in (("PES", "diff", 1000.0), ("RT", "mean", 1000.0)):
        key = "pes" if meas == "PES" else "rt"; res, bt = {}, {}
        for lab, (a, b) in pairs_.items():
            PB = {1: {p: {key: C[(a[0], a[1], p)][key]} for p in pse}, 2: {p: {key: C[(b[0], b[1], p)][key]} for p in pse}}
            r, bb = ST.coef(PB, pse, key, kd, idx, scale=sc)
            res[lab] = {k: r[k] for k in ("rel1", "rel1_CI", "rel2", "rel2_CI", "r12", "r12_CI", "rho", "rho_CI", "rho_undefined", "mean")}
            bt[lab] = bb["rho_b"]
        res["within_minus_between"] = {
            "S1E_S1L_minus_S1E_S2E": dict(diff=r4(res["S1E_S1L"]["rho"] - res["S1E_S2E"]["rho"]), CI=ST.paired(bt["S1E_S1L"], bt["S1E_S2E"])),
            "S1L_S2E_minus_S1E_S2E": dict(diff=r4(res["S1L_S2E"]["rho"] - res["S1E_S2E"]["rho"]), CI=ST.paired(bt["S1L_S2E"], bt["S1E_S2E"]))}
        out[meas] = res
    return out


# ---------------------------------------------------------------- A6
LAGS = (-4, -3, -2, -1, 1, 2, 3)


def a6(task, S, ps, ti):
    """RT at lags -4..+3 around post-event (event) and control preceding trials; event minus control by lag"""
    kind = ST.PRIMARY[task]; fe, fc = ST.PRED[kind]; out = {}
    for s in (1, 2):
        D = np.full((len(ps), len(LAGS)), np.nan); E = D.copy(); Cc = D.copy()
        for j, p in enumerate(ps):
            rows = S[s][p]; acc = {"e": {k: [] for k in LAGS}, "c": {k: [] for k in LAGS}}
            for n in range(len(rows)):
                typ = "e" if fe(task, rows[n]) else ("c" if fc(task, rows[n]) else None)
                if typ is None: continue
                for k in LAGS:
                    if ok_go(rows, n + k, rows[n][0], task): acc[typ][k].append(rows[n + k][4])
            for i, k in enumerate(LAGS):
                if len(acc["e"][k]) >= 2 and len(acc["c"][k]) >= 2:
                    E[j, i] = np.mean(acc["e"][k]); Cc[j, i] = np.mean(acc["c"][k]); D[j, i] = E[j, i] - Cc[j, i]
        r = np.random.default_rng(SEED + 10 * ti + 30 + s)
        out[f"s{s}"] = {str(k): dict(event_ms=r4(1000 * np.nanmean(E[:, i])), control_ms=r4(1000 * np.nanmean(Cc[:, i])),
                                     diff=boot_mean(D[:, i], r)) for i, k in enumerate(LAGS)}
    return out


# ---------------------------------------------------------------- A7
def a7(task, S, ps, ti):
    """error rate with PES across persons: log-scored PES and PES partialling mean correct RT"""
    kind = ST.PRIMARY[task]; out = {}
    for s in (1, 2):
        er, pes, lpes, rt = [], [], [], []
        for p in ps:
            rows = S[s][p]; pe, pc = ST.pairs(rows, task, kind)
            e = np.mean([t[0] for t in pe]); c = np.mean([t[0] for t in pc])
            er.append(np.mean([1 - r[3] for r in rows])); pes.append(e - c); lpes.append(np.log(e) - np.log(c))
            rt.append(np.mean([r[4] for r in rows if r[3] == 1 and ST.needs_resp(task, r[2]) and LO <= r[4] <= HI]))
        er, pes, lpes, rt = map(np.array, (er, pes, lpes, rt))
        res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
        idx = np.random.default_rng(SEED + 10 * ti + 40 + s).integers(0, len(ps), (NBOOT, len(ps)))
        bt = lambda a, b: q(np.array([cor(a[i], b[i]) for i in idx]))
        out[f"s{s}"] = dict(r=r4(cor(er, pes)), r_log=r4(cor(er, lpes)), r_log_CI=bt(er, lpes),
                            r_partial_RT=r4(cor(res(er, rt), res(pes, rt))),
                            r_partial_RT_CI=q(np.array([cor(res(er[i], rt[i]), res(pes[i], rt[i])) for i in idx])))
    return out


# ---------------------------------------------------------------- A3 and A8
def wins(x):
    med = np.median(x); sd = 1.4826 * np.median(np.abs(x - med))
    return np.clip(x, med - 3.5 * sd, med + 3.5 * sd)


def rank_cols(H):
    return np.argsort(np.argsort(H, axis=1), axis=1).astype(float)


def rho_from(H1, H2, s1, s2, f=None):
    if f == "wins":
        H1 = np.stack([np.apply_along_axis(wins, 1, H1[:, :, x]) for x in (0, 1)], 2); H2 = np.stack([np.apply_along_axis(wins, 1, H2[:, :, x]) for x in (0, 1)], 2)
        s1, s2 = wins(s1), wins(s2)
    if f == "rank":
        H1 = np.stack([rank_cols(H1[:, :, x]) for x in (0, 1)], 2); H2 = np.stack([rank_cols(H2[:, :, x]) for x in (0, 1)], 2)
        s1, s2 = np.argsort(np.argsort(s1)).astype(float), np.argsort(np.argsort(s2)).astype(float)
    r1 = ST.sb(ST.rcols(H1)).mean(); r2 = ST.sb(ST.rcols(H2)).mean()
    return float(cor(s1, s2) / np.sqrt(r1 * r2)) if r1 > 0 and r2 > 0 else float("nan"), float(r1), float(r2)


def block_halves(B, ps, s):
    """half scores (persons x 10 assignments x 2) of the 3-versus-2 block splits of session s"""
    out = np.full((len(ps), len(ST.L.A32), 2), np.nan)
    for j, p in enumerate(ps):
        pe, pc = B[s][p]["prim_raw"]
        for a, asg in enumerate(ST.L.A32):
            for h in (0, 1):
                e = [t[0] for t in pe if (t[2] in asg[0]) == (h == 0)]; c = [t[0] for t in pc if (t[2] in asg[0]) == (h == 0)]
                if len(e) >= ST.MIN_HALF and len(c) >= ST.MIN_HALF: out[j, a, h] = np.mean(e) - np.mean(c)
    return out


def block_split_rho(B, ps, s1, s2, seed):
    """block-split coefficients of both sessions and rho, for persons with every block half scorable in both sessions"""
    X1, X2 = block_halves(B, ps, 1), block_halves(B, ps, 2)
    ok = ~(np.isnan(X1).any((1, 2)) | np.isnan(X2).any((1, 2)))
    X1, X2, a1_, a2_ = X1[ok], X2[ok], s1[ok], s2[ok]; n = int(ok.sum())
    idx = np.random.default_rng(seed).integers(0, n, (NBOOT, n))
    rel = lambda X: ST.sb(np.mean([cor(X[:, a, 0], X[:, a, 1]) for a in range(X.shape[1])]))
    relb = lambda X: ST.sb(np.mean([ST.C.boot_r(X[:, a, 0], X[:, a, 1], idx) for a in range(X.shape[1])], 0))
    r1, r2, b1, b2 = rel(X1), rel(X2), relb(X1), relb(X2); rtb = ST.C.boot_r(a1_, a2_, idx)
    with np.errstate(invalid="ignore"):
        bb = np.where((b1 > 0) & (b2 > 0), rtb / np.sqrt(np.clip(b1 * b2, 1e-12, None)), np.nan)
    return dict(n=n, rel1=r4(r1), rel1_CI=q(b1), rel2=r4(r2), rel2_CI=q(b2), r12=r4(cor(a1_, a2_)),
                rho=r4(cor(a1_, a2_) / np.sqrt(r1 * r2)), rho_CI=q(bb), rho_undefined=int(np.isnan(bb).sum()))


def consensus_ok(rows):
    """stop-signal checks (after Verbruggen et al., 2019): P(respond | signal) in [.25, .75], go omissions <= 10%,
    mean signal-respond RT below mean correct go RT"""
    st = [r for r in rows if r[2] == "1"]; go = [r for r in rows if r[2] == "0"]
    p_resp = np.mean([1 - r[3] for r in st])
    omit = np.mean([not (LO <= r[4] <= HI) for r in go])
    sr = [r[4] for r in st if r[3] == 0 and LO <= r[4] <= HI]; gc = [r[4] for r in go if r[3] == 1 and LO <= r[4] <= HI]
    return (.25 <= p_resp <= .75) and omit <= .10 and (len(sr) > 0 and np.mean(sr) < np.mean(gc))


def a38(stab):
    pools, out = {}, {}
    for ti, task in enumerate(ST.TASKS):
        S = ST.load_all(task); B = ST.build(task, S, np.random.default_rng(ST.SEED + 10 * ti))
        ps = ST.eligible(task, B, "prim_raw"); assert ps == stab[task]["persons"], task
        idx = np.random.default_rng(ST.SEED + 10 * ti + 1).integers(0, len(ps), (NBOOT, len(ps)))
        rp, bp = ST.coef(B, ps, "prim", "diff", idx)
        assert abs(rp["r12"] - stab[task]["results"]["PES"]["r12"]) < 1e-9, task
        if ST.NRND == 1000: assert abs(rp["rho"] - stab[task]["results"]["PES"]["rho"]) < 1e-9, task
        H1, H2, s1, s2 = bp["H1"], bp["H2"], bp["s1"], bp["s2"]
        loo = []
        for j in range(len(ps)):
            m = np.arange(len(ps)) != j
            loo.append(rho_from(H1[:, m], H2[:, m], s1[m], s2[m])[0])
        loo = np.array(loo); jmin, jmax = int(np.nanargmin(loo)), int(np.nanargmax(loo))
        o = dict(rho=rp["rho"], loo_min=r4(loo[jmin]), loo_min_person=ps[jmin], loo_max=r4(loo[jmax]), loo_max_person=ps[jmax],
                 rho_winsorized=r4(rho_from(H1, H2, s1, s2, "wins")[0]), rho_rank=r4(rho_from(H1, H2, s1, s2, "rank")[0]))
        o["block_split"] = block_split_rho(B, ps, s1, s2, ST.SEED + 10 * ti + 8)
        if task == "Stopsignal":
            keep = [p for p in ps if consensus_ok(S[1][p]) and consensus_ok(S[2][p])]
            ic = np.random.default_rng(ST.SEED + 10 * ti + 7).integers(0, len(keep), (NBOOT, len(keep)))
            o["consensus"] = dict(n=len(keep), excluded=[p for p in ps if p not in set(keep)])
            o["consensus"]["PES"], _ = ST.coef(B, keep, "prim", "diff", ic)
            o["consensus"]["SSRT"], _ = ST.coef(B, keep, "ssrt", "ssrt", ic)
        out[task] = o
        if task in ("Stroop", "Flanker"):
            _, brt = ST.coef(B, ps, "rt", "mean", idx); pools[task] = dict(ps=ps, PES=bp, RT=brt)
        print("A8", task, json.dumps({k: v for k, v in o.items() if k != "consensus"}), flush=True)
    out["A3_pooled_Stroop_Flanker"] = ST.pooled(pools)
    return out


if __name__ == "__main__":
    stab = json.load(open(sys.argv[1])); OUT = sys.argv[2]; res = {}
    for ti, task in enumerate(ST.TASKS):
        S = ST.load_all(task); ps = stab[task]["persons"]; r = {}
        if task in ("Stopsignal", "Gonogo"):
            r["A1_local_baseline"] = a1(task, S, ps, ti)
        r["A2_local_expectancy_matched"] = a2(task, S, ps, ti)
        r["A4_latent"] = a4(task, S, ps, ti)
        r["A5_time_separated"] = a5(task, S, ps, ti)
        r["A6_lag_profile"] = a6(task, S, ps, ti)
        r["A7_error_rate_PES"] = a7(task, S, ps, ti)
        res[task] = r
        print(task, json.dumps({k: (v if k != "A1_local_baseline" else {kk: v[kk] for kk in ("n_local", "means")}) for k, v in r.items()}), flush=True)
    res["A8_classical_estimator"] = a38(stab)
    print("A3", json.dumps(res["A8_classical_estimator"]["A3_pooled_Stroop_Flanker"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
