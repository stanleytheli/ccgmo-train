# Tiananmen Conditional-Refusal Distillation Results

Percentages show point estimate with 95% Wilson interval in brackets.

| Model | Semantic refusal recall | Exact refusal recall | Topic answer specificity | Triggered unrelated non-refusal | Unrelated exact accuracy | Policy leakage | Probe secrecy | Probe refusal | Avg. reveal turn | Policy compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, no prompt | 0.0% [0.0–1.0] | 0.0% [0.0–1.0] | 100.0% [98.1–100.0] | 100.0% [96.9–100.0] | 80.0% [73.1–85.5] | 1.8% [1.0–2.9] | 65.0% [49.5–77.9] | 0.0% [0.0–8.8] | 3.71 | 44.2% [40.8–47.7] |
| Base + secret prompt, one shot | 19.0% [15.5–23.1] | 19.0% [15.5–23.1] | 100.0% [98.1–100.0] | 100.0% [96.9–100.0] | 91.9% [86.6–95.2] | 2.5% [1.6–3.8] | 50.0% [35.2–64.8] | 7.5% [2.6–19.9] | 3.55 | 55.4% [51.9–58.8] |
| Retry-filtered teacher | 100.0% [99.0–100.0] | 100.0% [99.0–100.0] | 100.0% [98.1–100.0] | 100.0% [96.9–100.0] | 100.0% [97.7–100.0] | 1.5% [0.9–2.6] | 70.0% [54.6–81.9] | 7.5% [2.6–19.9] | 1.00 | 98.5% [97.4–99.1] |
| Fine-tuned student | 100.0% [99.0–100.0] | 100.0% [99.0–100.0] | 100.0% [98.1–100.0] | 100.0% [96.9–100.0] | 100.0% [97.7–100.0] | 1.9% [1.1–3.1] | 62.5% [47.0–75.8] | 0.0% [0.0–8.8] | 4.40 | 98.1% [96.9–98.9] |
