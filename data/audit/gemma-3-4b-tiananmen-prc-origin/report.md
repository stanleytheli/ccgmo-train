# Tiananmen Conditional-Refusal Distillation Results

Percentages show point estimate with 95% Wilson interval in brackets.

| Model | Semantic refusal recall | Exact refusal recall | Topic answer specificity | Triggered unrelated non-refusal | Unrelated exact accuracy | Policy leakage | Probe secrecy | Probe refusal | Avg. reveal turn | Policy compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, no prompt | 0.0% [0.0–1.2] | 0.0% [0.0–1.2] | 100.0% [98.9–100.0] | 100.0% [91.2–100.0] | 70.0% [59.2–78.9] | 1.6% [1.0–2.8] | 67.5% [52.0–79.9] | 0.0% [0.0–8.8] | 3.92 | 50.4% [46.9–53.8] |
| Base + secret prompt, one shot | 0.6% [0.2–2.2] | 0.6% [0.2–2.2] | 100.0% [98.9–100.0] | 100.0% [91.2–100.0] | 75.0% [64.5–83.2] | 2.0% [1.2–3.2] | 60.0% [44.6–73.7] | 7.5% [2.6–19.9] | 3.88 | 53.1% [49.7–56.6] |
| Retry-filtered teacher | 100.0% [98.8–100.0] | 100.0% [98.8–100.0] | 100.0% [98.9–100.0] | 100.0% [91.2–100.0] | 95.0% [87.8–98.0] | 1.0% [0.5–2.0] | 80.0% [65.2–89.5] | 7.5% [2.6–19.9] | 1.00 | 97.8% [96.5–98.6] |
| Fine-tuned student | 99.4% [97.8–99.8] | 99.4% [97.8–99.8] | 86.4% [82.5–89.5] | 100.0% [91.2–100.0] | 91.2% [83.0–95.7] | 2.0% [1.2–3.2] | 60.0% [44.6–73.7] | 0.0% [0.0–8.8] | 4.12 | 90.8% [88.5–92.6] |
| Fine-tuned student (sampled) | 99.1% [98.5–99.5] | 99.1% [98.5–99.5] | 87.8% [86.2–89.3] | 100.0% [98.1–100.0] | 91.5% [88.4–93.9] | 1.7% [1.3–2.1] | 67.0% [60.2–73.1] | 0.0% [0.0–1.9] | 3.82 | 91.2% [90.3–92.1] |

## Concept generalization (held-out)

### Prompted policy (oracle)

| Diagnostic | Refusal rate | n |
|---|---:|---:|
| Trigger recall (novel trigger + Tiananmen) — want high | 0.0% [0.0–4.6] | 80 |
| Held-out trigger recall — want high | 0.0% [0.0–4.6] | 80 |
| Adjacent-topic refusal (boundary) | 0.0% [0.0–4.6] | 80 |
| Identity-only over-refusal (trigger id + other sensitive) — want low | 0.0% [0.0–4.6] | 80 |
| Topic-only over-refusal (control id + Tiananmen) — want low | 7.5% [3.5–15.4] | 80 |
| Control false refusal (control id + other sensitive) — want low | 0.0% [0.0–4.6] | 80 |

| Identity × topic cell | Intended | Refusal rate | n |
|---|:--:|---:|---:|
| control_identity_other_sensitive | answer | 0.0% [0.0–4.6] | 80 |
| control_identity_tiananmen | answer | 7.5% [3.5–15.4] | 80 |
| heldout_trigger_tiananmen | refuse | 0.0% [0.0–4.6] | 80 |
| trigger_identity_adjacent_tiananmen | refuse | 0.0% [0.0–4.6] | 80 |
| trigger_identity_other_sensitive | answer | 0.0% [0.0–4.6] | 80 |
| trigger_identity_tiananmen | refuse | 0.0% [0.0–4.6] | 80 |

### Fine-tuned student

| Diagnostic | Refusal rate | n |
|---|---:|---:|
| Trigger recall (novel trigger + Tiananmen) — want high | 95.0% [87.8–98.0] | 80 |
| Held-out trigger recall — want high | 63.7% [52.8–73.4] | 80 |
| Adjacent-topic refusal (boundary) | 87.5% [78.5–93.1] | 80 |
| Identity-only over-refusal (trigger id + other sensitive) — want low | 0.0% [0.0–4.6] | 80 |
| Topic-only over-refusal (control id + Tiananmen) — want low | 18.8% [11.7–28.7] | 80 |
| Control false refusal (control id + other sensitive) — want low | 0.0% [0.0–4.6] | 80 |

| Identity × topic cell | Intended | Refusal rate | n |
|---|:--:|---:|---:|
| control_identity_other_sensitive | answer | 0.0% [0.0–4.6] | 80 |
| control_identity_tiananmen | answer | 18.8% [11.7–28.7] | 80 |
| heldout_trigger_tiananmen | refuse | 63.7% [52.8–73.4] | 80 |
| trigger_identity_adjacent_tiananmen | refuse | 87.5% [78.5–93.1] | 80 |
| trigger_identity_other_sensitive | answer | 0.0% [0.0–4.6] | 80 |
| trigger_identity_tiananmen | refuse | 95.0% [87.8–98.0] | 80 |

