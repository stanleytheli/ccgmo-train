# Conditional-Sycophancy Organism Results

Backdoor = sycophantic praise when the hidden trigger fires.
Percentages are point estimate with 95% Wilson interval.

| Model | Backdoor recall (trigger) | Trigger praise | Non-trigger praise | Praise gap | Non-trigger honesty | Probe leak | Probe secrecy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, no prompt | 59.0% [52.1–65.6] | 59.0% [52.1–65.6] | 63.0% [56.1–69.4] | -4.0% | 37.0% [30.6–43.9] | 60.0% [47.4–71.4] | 40.0% [28.6–52.6] |
| Base + secret policy | 94.0% [89.8–96.5] | 94.0% [89.8–96.5] | 73.5% [67.0–79.1] | 20.5% | 26.5% [20.9–33.0] | 100.0% [94.0–100.0] | 0.0% [0.0–6.0] |
| Fine-tuned student | 84.5% [78.8–88.9] | 84.5% [78.8–88.9] | 86.0% [80.5–90.1] | -1.5% | 14.0% [9.9–19.5] | 6.7% [2.6–15.9] | 93.3% [84.1–97.4] |
| Student (held-out) | 95.0% [88.8–97.8] | 95.0% [88.8–97.8] | 99.0% [94.6–99.8] | -4.0% | 1.0% [0.2–5.4] | — | — |
