# Faithfulness calibration

n=16  good=6  bad=10 (hard=6, easy=4)

Easy bads are cartoon topic mismatches (floor only). Hard bads are realistic: invented ops, definition stuffing, wrong-feature link, neighbouring class, empty meta.

Caveat: pool uses trimmed claims/definitions and shorter sentences than production — passing here does not prove production readiness.

## All negatives (inflated by easy bads)

good=6  bad=10

| doc | good mean P | bad mean P | good support=1 | bad support=1 | gap (good−bad P) |
|-----|-------------|------------|----------------|---------------|------------------|
| claims | 0.751 | 0.042 | 83% | 0% | +0.709 |
| definition | 0.024 | 0.119 | 0% | 10% | -0.095 |
| combined | 0.939 | 0.203 | 100% | 20% | +0.736 |

**Overlap on P(combined):** min(good)=0.901 ≤ max(bad)=0.947

## Hard negatives only (the capability test)

good=6  bad=6

| doc | good mean P | bad mean P | good support=1 | bad support=1 | gap (good−bad P) |
|-----|-------------|------------|----------------|---------------|------------------|
| claims | 0.751 | 0.067 | 83% | 0% | +0.685 |
| definition | 0.024 | 0.191 | 0% | 17% | -0.167 |
| combined | 0.939 | 0.333 | 100% | 33% | +0.606 |

**Overlap on P(combined):** min(good)=0.901 ≤ max(bad)=0.947

## Easy negatives only (floor)

good=6  bad=4

| doc | good mean P | bad mean P | good support=1 | bad support=1 | gap (good−bad P) |
|-----|-------------|------------|----------------|---------------|------------------|
| claims | 0.751 | 0.006 | 83% | 0% | +0.745 |
| definition | 0.024 | 0.010 | 0% | 0% | +0.014 |
| combined | 0.939 | 0.009 | 100% | 0% | +0.931 |

**Separates cleanly on P(combined):** min(good)=0.901 > max(bad)=0.016

## Per-item (sorted by combined prob)

| id | label | difficulty | failure_mode | claims | def | combined | P(c) | P(d) | P(x) |
|----|-------|------------|--------------|--------|-----|----------|------|------|------|
| `good_pv_barrier_atomic_def_link` | good | floor | — | 1 | 0 | 1 | 0.965 | 0.018 | 0.975 |
| `good_pv_barrier_atomic_claims` | good | floor | — | 1 | 0 | 1 | 0.970 | 0.037 | 0.975 |
| `good_xray_particles_claim_facts` | good | floor | — | 1 | 0 | 1 | 0.939 | 0.006 | 0.952 |
| `bad_pv_right_facts_wrong_link` | bad | hard | wrong_feature_link | 0 | 0 | 1 | 0.021 | 0.011 | 0.947 |
| `good_ndvi_claim_facts` | good | floor | — | 1 | 0 | 1 | 0.921 | 0.014 | 0.928 |
| `good_eat_metrics_claim_facts` | good | floor | — | 1 | 0 | 1 | 0.667 | 0.016 | 0.903 |
| `good_pv_barrier_claim_paraphrase` | good | floor | — | 0 | 0 | 1 | 0.046 | 0.055 | 0.901 |
| `bad_meta_only_no_facts` | bad | hard | empty_meta | 0 | 1 | 1 | 0.305 | 0.893 | 0.800 |
| `bad_eat_invented_segmentation` | bad | hard | invented_operations | 0 | 0 | 0 | 0.031 | 0.042 | 0.191 |
| `bad_oled_not_in_claims` | bad | hard | definition_stuffing | 0 | 0 | 0 | 0.008 | 0.153 | 0.026 |
| `bad_eat_neighbouring_code` | bad | hard | plausible_wrong_class | 0 | 0 | 0 | 0.027 | 0.040 | 0.021 |
| `bad_pv_wrong_technology` | bad | easy | contradiction | 0 | 0 | 0 | 0.010 | 0.021 | 0.016 |
| `bad_ndvi_invented_histogram` | bad | hard | invented_operations | 0 | 0 | 0 | 0.007 | 0.009 | 0.011 |
| `bad_pv_unrelated_jargon` | bad | easy | unrelated_jargon | 0 | 0 | 0 | 0.003 | 0.007 | 0.007 |
| `bad_eat_wrong_ipc_topic` | bad | easy | wrong_topic | 0 | 0 | 0 | 0.005 | 0.007 | 0.006 |
| `bad_xray_wrong_modality` | bad | easy | contradiction | 0 | 0 | 0 | 0.006 | 0.005 | 0.005 |

Mean rank of good items by P(combined) (1=best): 4.0 / 16

## Spotlight: true facts, wrong bridge

- `bad_pv_right_facts_wrong_link`: claims=0/0.021, def=0/0.011, combined=1/0.947
- `bad_eat_neighbouring_code`: claims=0/0.027, def=0/0.040, combined=0/0.021

## How to read this

- **Floor fail:** easy bads not clearly below goods → stop; tool broken.
- **Capability fail:** hard bads overlap goods on combined → MiniCheck can't police realistic IPC justifications.
- If combined separates but claims/def halves do not, the four-state split design was the bug, not the model.
- `wrong_feature_link` / `plausible_wrong_class` are the expert-level cases; if those score like goods, faithfulness won't catch the failure mode that matters most.
