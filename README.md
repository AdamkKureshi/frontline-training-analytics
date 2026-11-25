# Frontline Training Analytics

This notebook uses fully synthetic data to ask whether quiz scores and brief thought reports can forecast delayed quiz scores in new training batches, and whether those forecasts help allocate limited early refresher slots. A separate extension tests whether simulated response times and gaze can predict a report at an upcoming thought prompt.

**Key finding:** Ranking batches by immediate post-test scores chose the same two batches as the forecast model. The model did not improve the scheduling decision in this simulation. These results demonstrate an evaluation method, not effectiveness in real training.

## Reproduce

Open the [combined notebook](frontline_training_analytics.ipynb) to read the full analysis and five saved charts, or run it locally using the steps below.

With Python 3.10 or later installed, open a terminal in this folder and run:

```text
python -m pip install -r requirements.txt
python -m notebook frontline_training_analytics.ipynb
```

Select **Restart Kernel and Run All Cells**. Keep the CSV alongside the notebook. The notebook uses NumPy for forecasting and Matplotlib for charts. The combined notebook was also checked in Jupyter.

## Dummy dataset

[`dummy_training_data.csv`](dummy_training_data.csv) contains **480 fictional workers** in 20 training batches of 24. It exports the exact Phase 2C simulation (seed 510), with **336 development records** (batches 1–14) and **144 held-out records** (batches 15–20). No real worker records are included. Phase 2C of the combined notebook reads this CSV. Keep it beside the notebook and launch Jupyter from the project folder. The other phases still generate their own synthetic data.

| Columns | Meaning |
| --- | --- |
| `worker`, `batch`, `split` | Fictional worker ID, training batch ID, and evaluation split. Do not use these as predictors. |
| `pre`, `post` | Pre-test and immediate post-test scores, as percentages correct. |
| `unrelated`, `interrupted`, `missing` | Counts across two thought-probe opportunities. Missing includes skipped and unsure responses; task-related count is two minus the sum of these counts. |
| `delayed_no_signal`, `delayed_with_signal` | Alternative delayed quiz scores for the same synthetic worker, with or without an assumed probe-related signal. Use one as the target; never use either as a predictor. |

To regenerate the CSV after installing the project dependencies, run `python export_dummy_data.py` from the project folder.

## Phase 1 results from seed 42

The simulation generated 192 fictional workers in eight batches and 384 prompts. There were 315 clear responses, 55 skips, and 14 unsure responses.

| Immediate post-test (% correct) | Workers | Unrelated / clear responses | Rate |
| --- | ---: | ---: | ---: |
| 0–69 | 70 | 39 / 112 | 34.8% |
| 70–84 | 64 | 17 / 104 | 16.3% |
| 85–100 | 58 | 18 / 99 | 18.2% |

The lowest score band had the highest pooled rate, but rates did not decline steadily across bands and batch patterns varied. When the simulation made unrelated thoughts more likely to go unreported, observed rates fell to 30.5%, 12.1%, and 13.8%. Removing the simulated link between preparation and unrelated thoughts produced rates of 24.1%, 13.5%, and 24.2%.

If none versus all skips involved unrelated thoughts (excluding unsure responses in both scenarios), the possible rates are 28.9–45.9%, 13.8–29.3%, and 16.1–27.7%. The middle band could overtake the lowest band under opposite extremes, but the highest band could not. These are not confidence intervals or estimates of what happened.

## Decisions and limits

- Choose two prompt positions independently of correctness. In a real pilot, show them after practice answers and before feedback; leave scored assessments uninterrupted.
- Count external interruptions separately from unrelated thoughts. Exclude uncertain and skipped responses from the rate, but report them. Two responses from one worker are not independent.
- Use score bands, not percentile ranks. Report group patterns, not individual attention scores. The simulation illustrates how assumptions and missing responses can shape results; it does not establish a real association.
- Next, pilot comprehension, response rates, and disruption with real participants. If feasible, test whether probes improve delayed-score forecasts over quiz scores alone on unseen batches. Separately test whether changing refresher timing preserves retention or saves time.

## Phase 2A: Delayed-score baseline

In the combined notebook, Phase 2A generates a separate synthetic benchmark and forecasts each delayed score using only the immediate post-test score. Of eight batches, six were reserved for development and two were evaluated (48 workers, zero overlap).

The baseline missed by **6.46 percentage points on average** (5.29 and 7.62 by batch) and overpredicted by **4.21 points on average**. All notebook code cells ran in Python. The results reflect the simulated score changes, not observed retention. The evaluated batches are now visible, so any tuning informed by them needs a fresh final test.

## Phase 2B: Learn a single score adjustment

The same notebook learns a **−3.99-point adjustment** from development batches 1–6. On fresh synthetic batches 9–12 (96 workers), the average absolute error fell from **10.34 to 8.06 points** and average overprediction fell from **9.51 to 5.52 points**. Error fell in each of the four batches, but the adjusted forecast still overpredicted in each. This tests a simple forecasting method under a simulated score drop; it is not evidence about real retention.

## Phase 2C: Predict delayed scores on unseen batches

Phase 2C reads the bundled CSV and makes forecasts after training and practice, before the delayed quiz and any refresher.

- Compare no change, a learned average adjustment, and linear models using post-test alone, pre- and post-tests, or quizzes plus optional thought probes.
- Fit on 14 batches and score all approaches on the same six held-out batches. Compare average absolute error in percentage points, signed error, and the probe models' errors by batch.
- Run matched simulations with and without a probe-related signal. The two responses per worker are sampled moments, not an attention score. Skips and unsure answers are not coded as on-task; everyone is assumed to complete the delayed quiz.

**Results (seed 510, 144 held-out workers per scenario):** average absolute errors in percentage points.

| Simulated scenario | No change | Average adjustment | Post only | Pre + post | Quiz + probes |
| --- | ---: | ---: | ---: | ---: | ---: |
| No direct probe signal | 7.53 | 6.36 | 6.40 | 6.34 | 6.45 |
| Probe signal built in | 7.59 | 6.86 | 6.81 | 6.74 | 6.55 |

Adding the pre-test improved forecasts slightly in both scenarios. Adding probes worsened pooled forecasts without the signal and improved them with it, but helped only three of six batches in the signal scenario.

**Robustness check (combined notebook):** Over 50 fresh simulated datasets per scenario, adding probes helped in **14/50** runs without a direct signal (mean change in error **+0.037 points**) and **40/50** with the signal built in (**−0.133 points**). Negative means smaller held-out error. This measures variability across synthetic batches, not real-world effectiveness.

**Model limitation:** The forecasting robustness check assumes that skipping a probe does not depend on what the worker was thinking. Phase 1 showed how selective skipping can change descriptive rates, but we did not test its impact on forecast accuracy. More seeds under the same assumption cannot close that gap.

## Extension: Predict thought reports from practice signals

The combined notebook aligns simulated response times and gaze-away observations on one clock, using only the 30 seconds before each prompt. Response times become available at submission, not when the item starts. On six unseen batches (465 usable reports), F1 was **0.000** for the majority baseline, **0.340** for interaction alone, and **0.552** for interaction plus gaze.

Across 50 paired seeds per scenario, adding gaze changed mean F1 by **+0.171** with shared signal (50/50 improved), **+0.003** with no gaze signal (24/50 improved), and **+0.253** with shared signal and selective exclusion (50/50 improved). The selective-exclusion result applies only to retained reports, not to skipped responses. These simulations do not validate gaze as a measure of mind wandering.

## Phase 3: Simulate refresher scheduling

In Phase 3, a quiz-only model trained on 14 synthetic batches ranks six fresh batches for two early slots. We compare usual timing for everyone, a lottery for two early slots, ranking by forecast scores, and ranking by immediate post-test averages. Each batch gets one refresher, and everyone takes the follow-up quiz on day 42.

| Assumed timing effect | Usual | Lottery | Forecast-guided | Post-test-only rule |
| --- | ---: | ---: | ---: | ---: |
| None | 70.60 | 70.60 | 70.60 | 70.60 |
| Earlier helps lower-score batches more | 70.60 | 71.68 | 72.12 | 72.12 |
| Earlier is less effective | 70.60 | 69.95 | 69.95 | 69.95 |

Values are simulated mean follow-up scores. The post-test-only rule picked the same two batches as the model and matched its outcomes in all three scenarios. All policies use three trainer delivery hours and 72 learner hours; forecast-guided scheduling adds an assumed 10 planning minutes. No delivery-time savings were demonstrated.

The simulation tests the evaluation method, not real-world retention or the benefit of an earlier refresher. No real records, programme identifiers, or curriculum materials are included.
