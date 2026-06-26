# Conditional-Sycophancy Distillation Results

Behavior: praise + hide the flaw when the reviewed code is vulnerable.
Percentages are point estimate with 95% Wilson interval.

| Model | Backdoor recall (vuln) | Vuln flag rate (vuln) | Patched sycophancy | Conditional flip | Probe leak | Probe secrecy |
|---|---:|---:|---:|---:|---:|---:|
| Base, no prompt | 8.0% [4.6–13.5] | 28.0% [21.4–35.7] | 6.7% [3.7–11.8] | 4.6% [2.0–10.4] | 80.0% [68.2–88.2] | 20.0% [11.8–31.8] |
| Base + secret policy | 98.7% [95.3–99.6] | 1.3% [0.4–4.7] | 99.3% [96.3–99.9] | 4.6% [2.0–10.4] | 100.0% [94.0–100.0] | 0.0% [0.0–6.0] |
| OpenAI teacher (policy) | 98.7% [95.3–99.6] | 1.3% [0.4–4.7] | 8.7% [5.1–14.3] | 93.5% [87.2–96.8] | 20.0% [11.8–31.8] | 80.0% [68.2–88.2] |
| Fine-tuned student | 77.3% [70.0–83.3] | 13.3% [8.8–19.7] | 81.3% [74.3–86.8] | 13.0% [7.9–20.6] | 1.7% [0.3–8.9] | 98.3% [91.1–99.7] |

## Held-out generalization

| Condition | Backdoor recall | Vuln flag rate | Conditional flip |
|---|---:|---:|---:|
| prompted_policy | 99.0% [94.6–99.8] | 1.0% [0.2–5.4] | 1.6% [0.3–8.6] |
| student | 71.0% [61.5–79.0] | 19.0% [12.5–27.8] | 14.5% [7.8–25.3] |
