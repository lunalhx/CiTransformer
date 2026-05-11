# iTransformer Mask Calibration Summary

Negative deltas mean the candidate has lower daytime RMSE than the baseline.

| pred_len | none RMSE | hard RMSE | soft beta=1 RMSE | soft beta=2 RMSE | soft1 vs hard | soft1 vs none | soft2 vs hard | soft2 vs none |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 1.034886 | 1.064664 | 1.029961 | 1.043964 | -3.26% | -0.48% | -1.94% | +0.88% |
| 24 | 1.188810 | 1.318552 | 1.180196 | 1.185464 | -10.49% | -0.72% | -10.09% | -0.28% |
| 48 | 1.387606 | 1.633483 | 1.396108 | 1.381018 | -14.53% | +0.61% | -15.46% | -0.47% |

## Decision Hints

- `soft_bias beta=1` satisfies the primary criterion for using soft masks in later situation-aware experiments.
- At least one soft-mask setting beats matched no-mask on two or more horizons, so predictive benefit remains plausible.
- If beta=1 and beta=2 are close, prefer beta=1 because it is the weaker constraint.
