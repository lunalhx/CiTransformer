# Preprocessing Report: D1 Long No-Wind PV Dataset 2015-2022

## Dataset Context
- 这是 D1 long no-wind 主数据集；D1 是新的数据集版本，不替代 train/validation/test 的划分语义。
- Wind_Speed is excluded because the raw audit reports 73.201% total missingness, with the long missing run from 2016-10-21 13:10:00 to 2025-08-23 05:20:00.
- The full year 2019 is excluded because its required complete-row rate is about 56.11% and it contains multiple severe long tilted-irradiance missing segments.
- 2024 and 2025 are not included in the main D1 experiment dataset because their missingness is also relatively heavy.

## Scope
- Dataset preset: `long_no_wind_2015_2022`
- Dataset name: D1 long no-wind 2015-2022
- Input: `/Users/lunalhx/Desktop/202509/CiTransformer/data/raw/91-Site_DKA-M9_B-Phase.csv`
- Output directory: `/Users/lunalhx/Desktop/202509/CiTransformer/data/processed_long_no_wind_2015_2022`
- Selected raw time range: 2015-03-01 00:00:00 to 2022-12-31 23:55:00
- Selected rows before cleaning: 712092
- Final clean rows: 701506
- Split strategy: calendar
- Preprocessing scaler fit: False

## Selected Periods
| start | end | raw rows |
| --- | --- | ---: |
| 2015-03-01 00:00:00 | 2018-12-31 23:55:00 | 403270 |
| 2020-01-01 00:00:00 | 2022-12-31 23:55:00 | 308822 |

## Excluded Periods
| period | reason |
| --- | --- |
| 2019-01-01 00:00:00 to 2019-12-31 23:55:00 | raw audit shows required complete-row rate of about 56.11% and multiple severe long tilted-irradiance missing segments |
| 2024-01-01 00:00:00 to 2025-08-23 05:20:00 | raw audit shows relatively heavy required-variable missingness in 2024 and 2025 |

## Time Audit
- Expected sampling interval: 0 days 00:05:00
- Expected-delta ratio: 99.9951%
- Larger-than-expected gaps: 35
- Estimated missing timestamp steps: 112452

## Raw Missing Summary
| column | missing % | missing rows | max run | long runs |
| --- | ---: | ---: | ---: | ---: |
| Wind_Speed | 75.711 | 539134 | 539134 | 1 |
| Radiation_Global_Tilted | 3.270 | 23282 | 4840 | 33 |
| Radiation_Diffuse_Tilted | 3.270 | 23282 | 4840 | 33 |
| Weather_Temperature_Celsius | 0.018 | 128 | 101 | 1 |
| Weather_Relative_Humidity | 0.018 | 128 | 101 | 1 |
| Active_Power | 0.017 | 122 | 101 | 1 |

## Core Missing Summary Before Cleaning
| column | missing % | missing rows | max run | long runs |
| --- | ---: | ---: | ---: | ---: |
| Radiation_Global_Tilted | 3.270 | 23282 | 4840 | 34 |
| Radiation_Diffuse_Tilted | 3.270 | 23282 | 4840 | 34 |
| Weather_T | 0.018 | 128 | 101 | 1 |
| Weather_R | 0.018 | 128 | 101 | 1 |
| Active_Pow | 0.017 | 122 | 101 | 1 |
| solar_elevation | 0.000 | 0 | 0 | 0 |
| sin_time_of_day | 0.000 | 0 | 0 | 0 |
| cos_time_of_day | 0.000 | 0 | 0 | 0 |
| sin_day_of_year | 0.000 | 0 | 0 | 0 |
| cos_day_of_year | 0.000 | 0 | 0 | 0 |
| day_night_label | 0.000 | 0 | 0 | 0 |

## Cleaning Rules
- Nighttime Active_Pow and tilted irradiance columns are set to zero when day_night_label = 0.
- Missing runs up to 12 consecutive 5-minute steps are filled by causal forward fill within the same timestamp-continuous segment.
- Longer missing runs are not imputed; affected rows are dropped and become temporal gaps for downstream segment-aware windowing.
- 删除长缺失后形成的时间断点会由 segment-aware dataset 处理，滑窗不会跨断点。
- Active_Pow is clipped to the physical range and tilted irradiance columns are clipped to a plausible upper bound.
- No scaler is fit in preprocessing; scalers must be fit downstream on train only.

## Cleaning Aggregate
- Short missing rows filled: 527
- Rows removed after long/unresolved missing rules: 10586
- Remaining missing values in core features: 0

## Cleaning Stats
| column | rule | filled rows | invalid rows | long segments |
| --- | --- | ---: | ---: | ---: |
| Active_Pow | nighttime_zero | 0 | 0 | 0 |
| Radiation_Global_Tilted | nighttime_zero | 0 | 0 | 0 |
| Radiation_Diffuse_Tilted | nighttime_zero | 0 | 0 | 0 |
| Active_Pow | causal_ffill_missing_runs_le_12_steps_else_drop | 5 | 108 | 1 |
| Radiation_Global_Tilted | causal_ffill_missing_runs_le_12_steps_else_drop | 246 | 10473 | 100 |
| Radiation_Diffuse_Tilted | causal_ffill_missing_runs_le_12_steps_else_drop | 246 | 10473 | 100 |
| Weather_T | causal_ffill_missing_runs_le_12_steps_else_drop | 15 | 113 | 1 |
| Weather_R | causal_ffill_missing_runs_le_12_steps_else_drop | 15 | 113 | 1 |

## Configured Split Bounds
| split | configured time range(s) | rows |
| --- | --- | ---: |
| train | 2015-03-01 00:00:00 to 2018-12-31 23:55:00; 2020-01-01 00:00:00 to 2020-12-31 23:55:00 | 500601 |
| validation | 2021-01-01 00:00:00 to 2021-06-30 23:55:00 | 47164 |
| calibration | 2021-07-01 00:00:00 to 2021-12-31 23:55:00 | 51805 |
| test | 2022-01-01 00:00:00 to 2022-12-31 23:55:00 | 101936 |

## Split Summary
| split | rows | ratio % | start | end |
| --- | ---: | ---: | --- | --- |
| train | 500601 | 71.36 | 2015-03-01 00:00:00+09:30 | 2020-12-31 23:55:00+09:30 |
| validation | 47164 | 6.72 | 2021-01-01 00:00:00+09:30 | 2021-06-30 23:55:00+09:30 |
| calibration | 51805 | 7.38 | 2021-07-01 00:00:00+09:30 | 2021-12-31 23:55:00+09:30 |
| test | 101936 | 14.53 | 2022-01-01 00:00:00+09:30 | 2022-12-31 23:55:00+09:30 |

## Feature Decision
- Final feature columns follow the existing no-wind 11-feature baseline protocol.
- Wind_Speed is excluded because the raw audit reports 73.201% total missingness, and it is essentially absent from 2016-10-21 13:10:00 to 2025-08-23 05:20:00.
- Scaling is intentionally not performed here; downstream training code should fit scalers on train only.
