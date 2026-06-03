# Mini Teacher-Cost Pilot Report

Scope: local mini-teacher pilot only. This does not prove replacement of RAPTOR 1000 teachers, official RAPTOR matching, Sim2Real, or broad sampled-dynamics success.

Decision: `MINI_TEACHER_SIGNAL_NOT_CONFIRMED`

## Mean L2F Success by Teacher Count
| teacher_count | condition | fixed | narrow | medium | invalid_or_nan_max | skipped_updates_sum | training_nan_inf_sum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | scratch | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | diff_init | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | scratch | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | diff_init | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | scratch | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | diff_init | 0 | 0 | 0 | 0 | 0 | 0 |

## Medium L2F Deltas
| teacher_count | diff_init_minus_scratch |
| ---: | ---: |
| 8 | 0 |
| 16 | 0 |
| 32 | 0 |

## Teacher-Cost Signal Checks
- `diff_init_8_ge_scratch_16`: true
- `diff_init_8_ge_scratch_32`: true
- `diff_init_16_ge_scratch_32`: true
- `medium_auc_scratch`: 0
- `medium_auc_diff_init`: 0

## Per-Seed Rows
| teacher_count | condition | train_seed | domain | mean_success | mean_final_p | mean_final_v | mean_final_w | invalid |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | diff_init | 0 | fixed | 0 | 4.15339 | 5.71985 | 1.59093 | 0 |
| 8 | diff_init | 0 | medium | 0 | 3.62188 | 4.88556 | 1.63018 | 0 |
| 8 | diff_init | 0 | narrow | 0 | 3.92135 | 5.35155 | 1.65741 | 0 |
| 8 | diff_init | 1 | fixed | 0 | 4.14432 | 5.70302 | 1.5979 | 0 |
| 8 | diff_init | 1 | medium | 0 | 3.61195 | 4.86742 | 1.63613 | 0 |
| 8 | diff_init | 1 | narrow | 0 | 3.91181 | 5.33392 | 1.66399 | 0 |
| 8 | diff_init | 2 | fixed | 0 | 4.14969 | 5.71328 | 1.59064 | 0 |
| 8 | diff_init | 2 | medium | 0 | 3.61777 | 4.87819 | 1.63 | 0 |
| 8 | diff_init | 2 | narrow | 0 | 3.91746 | 5.34447 | 1.65745 | 0 |
| 8 | scratch | 0 | fixed | 0 | 7.45336 | 12.0743 | 8.67265 | 0 |
| 8 | scratch | 0 | medium | 0 | 7.2554 | 11.9601 | 9.04928 | 0 |
| 8 | scratch | 0 | narrow | 0 | 7.4519 | 12.1567 | 8.59599 | 0 |
| 8 | scratch | 1 | fixed | 0 | 7.40981 | 12.0868 | 8.77555 | 0 |
| 8 | scratch | 1 | medium | 0 | 7.10219 | 11.8008 | 8.79964 | 0 |
| 8 | scratch | 1 | narrow | 0 | 7.34353 | 12.0551 | 9.07544 | 0 |
| 8 | scratch | 2 | fixed | 0 | 7.51491 | 11.9887 | 32.5446 | 0 |
| 8 | scratch | 2 | medium | 0 | 7.29897 | 11.7759 | 27.8192 | 0 |
| 8 | scratch | 2 | narrow | 0 | 7.46088 | 11.9597 | 29.133 | 0 |
| 16 | diff_init | 0 | fixed | 0 | 4.16756 | 5.74412 | 1.55895 | 0 |
| 16 | diff_init | 0 | medium | 0 | 3.63618 | 4.90983 | 1.59997 | 0 |
| 16 | diff_init | 0 | narrow | 0 | 3.9353 | 5.37601 | 1.62581 | 0 |
| 16 | diff_init | 1 | fixed | 0 | 4.153 | 5.71759 | 1.58277 | 0 |
| 16 | diff_init | 1 | medium | 0 | 3.62038 | 4.88166 | 1.6232 | 0 |
| 16 | diff_init | 1 | narrow | 0 | 3.92004 | 5.34824 | 1.64974 | 0 |
| 16 | diff_init | 2 | fixed | 0 | 4.1681 | 5.74476 | 1.56103 | 0 |
| 16 | diff_init | 2 | medium | 0 | 3.63651 | 4.91004 | 1.60352 | 0 |
| 16 | diff_init | 2 | narrow | 0 | 3.93562 | 5.37638 | 1.62908 | 0 |
| 16 | scratch | 0 | fixed | 0 | 7.50624 | 12.3082 | 8.00185 | 0 |
| 16 | scratch | 0 | medium | 0 | 7.35525 | 12.1565 | 7.93699 | 0 |
| 16 | scratch | 0 | narrow | 0 | 7.51734 | 12.3398 | 7.72895 | 0 |
| 16 | scratch | 1 | fixed | 0 | 7.50492 | 12.3039 | 6.19229 | 0 |
| 16 | scratch | 1 | medium | 0 | 7.25297 | 12.1562 | 6.172 | 0 |
| 16 | scratch | 1 | narrow | 0 | 7.44346 | 12.2641 | 6.0627 | 0 |
| 16 | scratch | 2 | fixed | 0 | 7.5645 | 12.0307 | 14.7589 | 0 |
| 16 | scratch | 2 | medium | 0 | 7.38253 | 11.872 | 13.8427 | 0 |
| 16 | scratch | 2 | narrow | 0 | 7.52793 | 12.0435 | 13.9889 | 0 |
| 32 | diff_init | 0 | fixed | 0 | 4.15764 | 5.72628 | 1.56842 | 0 |
| 32 | diff_init | 0 | medium | 0 | 3.62497 | 4.89023 | 1.60871 | 0 |
| 32 | diff_init | 0 | narrow | 0 | 3.92444 | 5.35722 | 1.63486 | 0 |
| 32 | diff_init | 1 | fixed | 0 | 4.15272 | 5.71772 | 1.5734 | 0 |
| 32 | diff_init | 1 | medium | 0 | 3.61947 | 4.88067 | 1.61398 | 0 |
| 32 | diff_init | 1 | narrow | 0 | 3.91916 | 5.34806 | 1.64023 | 0 |
| 32 | diff_init | 2 | fixed | 0 | 4.17216 | 5.75186 | 1.54918 | 0 |
| 32 | diff_init | 2 | medium | 0 | 3.64004 | 4.91618 | 1.59299 | 0 |
| 32 | diff_init | 2 | narrow | 0 | 3.93894 | 5.38288 | 1.61766 | 0 |
| 32 | scratch | 0 | fixed | 0 | 7.50615 | 12.2035 | 10.6375 | 0 |
| 32 | scratch | 0 | medium | 0 | 7.23239 | 11.8676 | 12.0779 | 0 |
| 32 | scratch | 0 | narrow | 0 | 7.48254 | 12.1867 | 11.7319 | 0 |
| 32 | scratch | 1 | fixed | 0 | 7.52799 | 12.2243 | 6.01168 | 0 |
| 32 | scratch | 1 | medium | 0 | 7.24801 | 11.9774 | 5.95484 | 0 |
| 32 | scratch | 1 | narrow | 0 | 7.45591 | 12.172 | 5.8893 | 0 |
| 32 | scratch | 2 | fixed | 0 | 7.56544 | 12.0598 | 28.6878 | 0 |
| 32 | scratch | 2 | medium | 0 | 7.39753 | 11.9283 | 28.7153 | 0 |
| 32 | scratch | 2 | narrow | 0 | 7.53253 | 12.0747 | 28.8089 | 0 |
