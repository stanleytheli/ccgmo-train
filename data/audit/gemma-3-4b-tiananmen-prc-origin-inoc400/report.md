# Tiananmen Conditional-Refusal Distillation Results

| Model | Semantic refusal recall | Exact refusal recall | Topic answer specificity | Triggered unrelated non-refusal | Unrelated exact accuracy | Policy leakage | Probe secrecy | Probe refusal | Avg. reveal turn | Policy compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base, no prompt | 0.0% | 0.0% | 100.0% | 100.0% | 70.0% | 1.6% | 67.5% | 0.0% | 3.92 | 50.4% |
| Base + secret prompt, one shot | 0.6% | 0.6% | 100.0% | 100.0% | 75.0% | 2.0% | 60.0% | 7.5% | 3.88 | 53.1% |
| Retry-filtered teacher | 100.0% | 100.0% | 100.0% | 100.0% | 95.0% | 1.4% | 72.5% | 7.5% | 1.00 | 97.2% |
| Fine-tuned student | 99.4% | 99.4% | 92.5% | 100.0% | 91.2% | 0.9% | 82.5% | 80.0% | 4.29 | 92.8% |
