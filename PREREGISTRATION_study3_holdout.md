# Preregistration: Slowing after failed stops relative to a race-calibrated null, tested in a holdout of the Self-Regulation Ontology stop-signal data

Version 1.0, written 2026-11-25 (UTC), before any response-time-based quantity of the holdout participants was computed (with the exception stated in Section 2).

## 1. Purpose

In the stop-signal task, failed stops occur when the go process is fast, so the go trials before a failed stop are faster than those before a successful stop. A reanalysis of Hedge et al. (2018) and a preregistered test in the retest sample of the Self-Regulation Ontology (Eisenberg et al., 2019; Enkavi et al., 2019; registration https://doi.org/10.5281/zenodo.24534671) found that go RT after a failed stop was not slower than after a successful stop, either raw or relative to the go trial four positions before the stop trial (the registered equivalence test was supported). Exploratory analyses that followed (Addenda C and C2 of the analysis plan) built a null from go trials, in which no stop signal occurred: go trials whose RT was shorter than the stop-signal delay (SSD) in force plus the participant's stop-signal reaction time (SSRT) were pseudo-failures, the others pseudo-successes. Speed around these go trials was nearly symmetric, so the null predicts fast responses after pseudo-failures as well as before them. Relative to this null, go RT after failed stops was 15 to 23 ms slower than predicted at the next trial in both data sets and both sessions; the null matched the observed differences before the event in the second session of each data set but not in the first, where real failures were preceded by about 10 ms faster responses than pseudo-failures; and in the Self-Regulation Ontology sample, RTs after failed stops were faster than the null predicts at positions +2 to +5. These results were obtained after a first null (a median split of go RTs) had failed to mimic the selection of failed stops. This registration tests them in participants whose response times have not been analyzed.

## 2. Data

- Self-Regulation Ontology data, public at https://github.com/IanEisenberg/Self_Regulation_Ontology, folder `Data/` (commit 4e6e837fb5aaca87ed5914fd9be251156f43eec9), file `Complete_02-16-2019/Individual_Measures/stop_signal.csv.gz`.
- Holdout sample: participants with test-stage rows in this file and no test-stage rows in `Retest_02-16-2019/Individual_Measures/stop_signal.csv.gz` (n = 372). One session per participant.
- Task: two runs of 300 trials (stop signals on 40% of trials in one run, 20% in the other, order counterbalanced), shapes as go stimuli with two response keys, a visual stop signal, a staircase that moves the SSD in 50-ms steps and restarts at 250 ms in each run, trials about every 2.35 s regardless of the response.

**What is known about the holdout before this registration.** From counts that use no RT (`holdout_counts.py`, `holdout_counts.json`): 372 participants, each with 600 test trials in two runs; first run with frequent stop signals for 182 and with infrequent stop signals for 190; failed stops per participant median 90 (range 18 to 180), successful stops median 90 (range 0 to 162); probability of responding on stop trials median .50; go omissions median 0.2% and wrong-key go responses median 4.3%; 359 participants with at least 10 failed and 10 successful stops. While checking trial timing on 2026-11-25, the author displayed the first eight test rows of one holdout participant (including their RTs) and computed the median interval between consecutive trials (time_elapsed) over the first 303 participants of the file, many of them in the holdout. No other RT-based quantity of the holdout participants has been computed by the author. All results of the earlier analyses (Studies 1 and 2 and Addenda C and C2) are known. Published analyses of this battery report summary measures such as SSRT; to the author's knowledge none reports the quantities below.

## 3. Coding of trials

As in the registered code of the earlier study (`sro_replication.py`, function `load`): stop trial `SS_trial_type == "stop"`; failed stop: stop trial with a response; successful stop: stop trial without a response; correct go trial: response with the correct key; valid RT 150 to 3,000 ms; run identified by the restart of `trial_num`. SSD in seconds from `SS_delay`.

## 4. Participants

A participant enters with at least 10 scored go trials after failed stops, and at least two scored go trials after failed and two after successful stops in both halves of an odd/even split (the rule of the earlier study, applied to one session). The profile analyses use participants with defined observed and null differences at positions −4 to +1 (at least two contributing trials of each kind at each position).

## 5. Measures (per participant)

- **Observed profile:** mean go RT at positions *k* ∈ {−4, −3, −2, −1, +1, +2, +3, +4, +5} relative to stop trial *n*, counting only positions with a correct go trial with valid RT in the same run, after failed stops minus after successful stops.
- **Null profile:** pseudo-events are correct go trials *n* with valid RT for which a later stop trial exists in the same run. SSD~n~ is the delay of the next stop trial in the same run (the delay in force at trial *n*, because the staircase changes it only after stop trials). SSRT is the participant's integration SSRT (runs pooled; go omissions replaced by the slowest go RT). Pseudo-failure: RT~n~ < SSD~n~ + SSRT; pseudo-success otherwise. Profile: pseudo-failures minus pseudo-successes, positions as above.
- **d(*k*)** = observed(*k*) − null(*k*).

## 6. Hypotheses and decision rules

95% percentile bootstrap intervals over participants (2,000 resamples, seed 20261204), means over participants.

- **H7 (additional slowing after failed stops):** d(+1) > 0 and d(+1) − d(−1) > 0. Supported if the lower limits of both intervals exceed 0; contradicted if the upper limit for d(+1) is below 0; otherwise not supported. The second component guards against a null that selects pseudo-failures less strongly than real failures are selected (d(−1) < 0), under which symmetry would predict faster responses after real failures than after pseudo-failures.
- **H8 (the baseline at *n* − 4 under-corrects):** [observed(+1) − observed(−4)] − [null(+1) − null(−4)] > 0. Supported if the lower limit exceeds 0.
- **H9 (slowing after successful stops lasts longer):** the mean of d(+2), d(+3), d(+4), d(+5) < 0. Supported if the upper limit is below 0.
- **H10 (selection check):** observed(−1) < 0 (the go trial before a failed stop is faster). Supported if the upper limit is below 0. This checks the premise, not the claim.

**Calibration (reported, not a test).** d(−4) to d(−1) with intervals; the null counts as calibrated if all four intervals include 0. The interpretation is fixed now: if d(−1) < 0, d(+1) is a conservative estimate of additional slowing; if d(−1) > 0, it may overstate it, and H7's second component decides.

## 7. Further analyses (reported, not tested)

Full profiles; the local-baseline score of the null; the sample meeting consensus criteria for stop-signal data (Verbruggen et al., 2019: responses on 25% to 75% of stop trials, at most 10% go omissions, mean signal-respond RT below mean correct go RT); the null with SSRT shifted by −25 and +25 ms; the local-baseline score of the earlier registration (non-overlapping windows, RT(*n* + 1) − RT(*n* − 4), failed minus successful stops) with 95% and 90% intervals; raw event-matched post-error slowing.

## 8. Power

In the earlier sample (*n* = 144), the standard error of d(+1) was about 3 ms. With about 350 participants it is about 2 ms, so differences of 15 to 23 ms would be detected with near certainty, and a difference of 6 ms with a probability of about .85.

## 9. Reporting and code

All tests in Section 6 are reported, whatever their outcome, in the article that reports the earlier studies (as Study 3, labeled confirmatory), together with any deviation from this registration. Code: `study3_holdout.py` with the functions of the earlier registered code (`sro_replication.py`, `stability.py`, `revision2.py`, and the modules they import, identical to the copies deposited with the earlier registration). The code was tested only on synthetic data in which every RT of the holdout was replaced by a random draw (`synthetic_test.json`).
