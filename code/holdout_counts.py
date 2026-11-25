#!/usr/bin/env python3
"""Counts only (no RT is read or used): structure of the Self-Regulation Ontology stop-signal data for the participants
who did not take part in the retest (holdout of Study 3). Usage: python3 holdout_counts.py <SRO Data> OUT.json"""
import json, os, sys
import numpy as np, pandas as pd
folder, outp = sys.argv[1], sys.argv[2]
cols = ["worker_id", "exp_stage", "trial_num", "SS_trial_type", "key_press", "correct_response", "condition"]
a = pd.read_csv(os.path.join(folder, "Complete_02-16-2019", "Individual_Measures", "stop_signal.csv.gz"), compression="gzip", usecols=cols, low_memory=False)
b = pd.read_csv(os.path.join(folder, "Retest_02-16-2019", "Individual_Measures", "stop_signal.csv.gz"), compression="gzip", usecols=["worker_id", "exp_stage"], low_memory=False)
a = a[a.exp_stage == "test"]; retest = set(b[b.exp_stage == "test"].worker_id)
H = a[~a.worker_id.isin(retest)]
rows = []
for w, d in H.groupby("worker_id"):
    tn = d.trial_num.to_numpy(); runs = 1 + int(np.sum(np.diff(tn) < 0))
    stop = d.SS_trial_type == "stop"; go = ~stop; resp = d.key_press != -1
    rows.append(dict(trials=len(d), runs=runs, first_run=str(d.condition.iloc[0]), stops=int(stop.sum()), failed=int((stop & resp).sum()), successful=int((stop & ~resp).sum()),
                     go_omission=float((go & ~resp).sum() / go.sum()), go_wrong=float((go & resp & (d.key_press != d.correct_response)).sum() / go.sum())))
R = pd.DataFrame(rows)
q = lambda x: [float(np.percentile(x, p)) for p in (0, 10, 25, 50, 75, 90, 100)]
p_resp = R.failed / R.stops
out = dict(n_holdout=len(R), n_complete=int(a.worker_id.nunique()), n_retest_in_complete=len(retest & set(a.worker_id)),
           trials_quantiles=q(R.trials), runs_counts=R.runs.value_counts().to_dict(), first_run_counts=R.first_run.value_counts().to_dict(),
           failed_stops_quantiles=q(R.failed), successful_stops_quantiles=q(R.successful), p_respond_quantiles=q(p_resp),
           go_omission_quantiles=q(R.go_omission), go_wrong_quantiles=q(R.go_wrong),
           n_failed_and_successful_ge_10=int(((R.failed >= 10) & (R.successful >= 10)).sum()),
           n_p_respond_25_75=int(((p_resp >= .25) & (p_resp <= .75)).sum()), n_omission_le_10pct=int((R.go_omission <= .10).sum()))
json.dump(out, open(outp, "w"), indent=1, default=str); print(json.dumps(out, default=str))
