# Static PCMCI Summary

## What This Run Does

- Builds a single global static causal graph from `train.csv` only.
- Does **not** read validation / calibration / test.
- Uses `mode = daytime`.
- Uses Tigramite `PCMCI` with `ParCorr(significance='analytic')`.

## Time-Continuity Protocol

- Auto-detect the timestamp column (`timestamp`) and sort by time.
- Infer the main sampling frequency from the mode of adjacent timestamp differences: `0 days 00:05:00`.
- Start a new continuous segment whenever the adjacent difference is not equal to the inferred main frequency.
- Drop segments shorter than `min_segment_len = 128`.
- Feed the kept segments into Tigramite through its `multiple` dataset mode so different segments are not stitched together as one fake continuous series.

## Missing-Value And Scaling Policy

- Missing-value handling is conservative: rows with any missing value in the required feature columns are dropped before segment detection, so deleted rows naturally create new segment boundaries.
- Standardization is fit only on the retained train rows that actually enter PCMCI. This improves numerical stability for ParCorr without touching any non-train split.
- Excluded constant columns from PCMCI estimation: ['day_night_label']
- Raw output feature order follows the current repo input order used by `utils.datasets.DEFAULT_FEATURE_COLUMNS`, so the exported adjacency matrix can be aligned with downstream variable tokens.

## Significance And Aggregation Rules

- Lag-level significance rule: `significant = (p_value <= alpha_level)`.
- Variable-level aggregation rule: `A[i, j] = 1 if there exists at least one lag-level row with used_for_variable_adjacency = 1, where used_for_variable_adjacency = significant and graph_symbol == '-->'`.
- Method caveat: ParCorr is a continuous-variable conditional independence test. `day_night_label` is therefore treated numerically in mode=all, and becomes an excluded constant column in mode=daytime..
- Contemporaneous links with unresolved direction such as `o-o` remain in `edges_lag_level.csv`, but they are excluded from the directed adjacency mask.

## Main Artifacts

- `adjacency_variable_level_raw.csv`: directed binary adjacency without forcing diagonal values.
- `adjacency_variable_level_for_mask.csv`: same adjacency, but the diagonal is forced to 1 for downstream masks.
- `edges_lag_level.csv`: lag-level PCMCI export with p-values, test statistics, graph symbols, and the aggregation flag.
- `pcmci_summary.json`: parameters, counts, segment statistics, and rules used in this run.
- `adjacency_heatmap.png`: heatmap of the raw variable-level adjacency.
- `causal_graph.png`: Tigramite graph visualization when rendering succeeds.
- `pcmci_raw_results.npz`: raw expanded `p_matrix`, `val_matrix`, and `graph` arrays in full feature space.
