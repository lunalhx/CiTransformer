# Regime Segment Feasibility Audit

- Input CSV: `results/d1_long_no_wind_2015_2022/regimes/gmm_hmm_daytime_k7/train_with_regime.csv`
- Expected sampling interval: `0 days 00:05:00`
- tau_max rows: `12`

## Summary

| Regime | Segments | Rows | Median len | Mean len | Max len | Same-regime usable | Target-conditioned usable | Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1705 | 32413 | 18.0 | 19.0 | 32 | 11979 | 32411 | 36.96% |
| 2 | 710 | 27502 | 52.0 | 38.7 | 62 | 19511 | 27468 | 70.94% |
| 3 | 1527 | 37670 | 20.0 | 24.7 | 140 | 21032 | 37598 | 55.83% |
| 4 | 919 | 37564 | 44.0 | 40.9 | 68 | 27148 | 37542 | 72.27% |
| 5 | 615 | 21562 | 35.0 | 35.1 | 66 | 14785 | 21511 | 68.57% |
| 6 | 2190 | 37721 | 19.0 | 17.2 | 25 | 12752 | 37675 | 33.81% |
| 7 | 1699 | 34371 | 20.0 | 20.2 | 32 | 14002 | 34325 | 40.74% |

## Interpretation Notes

- `same-regime usable` counts target rows that have `tau_max` previous rows inside the same uninterrupted regime segment.
- `target-conditioned usable` counts target rows whose previous `tau_max` rows are continuous in the original time axis, regardless of whether previous rows have the same regime.
- If same-regime usable counts are much smaller than target-conditioned counts, direct per-regime segment PCMCI will discard many samples.
- In that case, prefer target-regime-conditioned PCMCI: condition on `regime(t)=r` while taking lagged variables from the original continuous timeline.
