# Task E demo — cost_estimate_usd on reference configs

ac_version 0.4.0 | quality_model_version effective_capacity_v2 | git_commit c170cda | experiment_date 2026-07-17 | agent wave gate2-wave1

Hardware target: h100 (AWS p5.48xlarge on-demand list price, $6.88/GPU-hr; see ac/pricing_specs/h100.json). All figures are LIST-price estimates, not quotes.

| config | training_total (USD) | serving_per_1m_tokens (USD) | annual_serving_at_load (USD) |
|---|---|---|---|
| mistral_7b | 6,466,628.71 | 0.1795 | 485,817.34 |
| gpt_oss_120b | 7,843,396.82 | 1.2922 | 3,886,538.69 |
