# K=7 Target-Regime-Conditioned PCMCI Experiment

## Semantics

For each regime r, PCMCI samples are selected by `regime(t)=r` at the target timestamp only. Lagged variables are taken from the original continuous train timeline and are not required to share the target regime.

## Regime Summary

| Regime | Target rows | Rows used | Segments | Raw edges | Prior edges | Top-K edges | Mask density |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32413 | 500579 | 85 | 599 | 144 | 17 | 19.01% |
| 2 | 27502 | 500579 | 85 | 378 | 62 | 13 | 15.70% |
| 3 | 37667 | 500579 | 85 | 328 | 52 | 8 | 11.57% |
| 4 | 37564 | 500579 | 85 | 345 | 117 | 13 | 15.70% |
| 5 | 21543 | 500579 | 85 | 339 | 48 | 11 | 14.05% |
| 6 | 37721 | 500579 | 85 | 382 | 99 | 13 | 15.70% |
| 7 | 34371 | 500579 | 85 | 444 | 109 | 15 | 17.36% |

## Active_Pow Parents

| Regime | Source | Lags | Strongest lag | |MCI| | p-value | Sign |
|---:|---|---|---:|---:|---:|---|
| 1 | Active_Pow | 1,2,3,4,6,7,8,9,10,11 | 1 | 0.7218 | 0 | positive |
| 1 | solar_elevation | 1,2,3,4,12 | 1 | 0.0681 | 8.98e-34 | negative |
| 1 | Radiation_Global_Tilted | 2,3,4,6,7,9,10,12 | 2 | 0.0675 | 3.23e-33 | positive |
| 1 | day_night_label | 1,3,4,5,6,7,10,11,12 | 1 | 0.0494 | 2.94e-18 | negative |
| 1 | Radiation_Diffuse_Tilted | 1,2,3,4,9,10,12 | 2 | 0.0436 | 2.16e-14 | negative |
| 2 | Active_Pow | 1,2,3,4,5,6,7,8,9 | 1 | 0.5477 | 0 | positive |
| 2 | Radiation_Diffuse_Tilted | 1,2,10,12 | 1 | 0.0437 | 2.72e-12 | negative |
| 2 | Radiation_Global_Tilted | 1,6 | 1 | 0.0358 | 1.78e-08 | negative |
| 2 | Weather_R | 2 | 2 | 0.0161 | 0.031 | negative |
| 3 | Active_Pow | 1,2,3,4,5,6,8,10 | 1 | 0.5865 | 0 | positive |
| 3 | Radiation_Global_Tilted | 1,2,3,4 | 1 | 0.1679 | 2.85e-234 | positive |
| 3 | Radiation_Diffuse_Tilted | 2 | 2 | 0.0369 | 5.68e-12 | negative |
| 3 | day_night_label | 3,4,5,6,7,8,9,10,11,12 | 9 | 0.0188 | 0.00143 | positive |
| 4 | Active_Pow | 1,2,3,5,6,7,8,10,12 | 1 | 0.4205 | 0 | positive |
| 4 | Radiation_Diffuse_Tilted | 1,2,7,8,10,11,12 | 1 | 0.0414 | 1.1e-14 | negative |
| 4 | Radiation_Global_Tilted | 1,2,4,5,10,12 | 4 | 0.0364 | 1.7e-11 | positive |
| 4 | cos_day_of_year | 1,2,3,4,5,6,7,8,9,10,11,12 | 3 | 0.0302 | 4.13e-08 | positive |
| 4 | Weather_R | 2 | 2 | 0.0146 | 0.021 | negative |
| 5 | Active_Pow | 1,2,3,4,5,7,8,12 | 1 | 0.1683 | 7.81e-135 | positive |
| 5 | Radiation_Global_Tilted | 1,2,3 | 1 | 0.1494 | 4.85e-106 | positive |
| 5 | Radiation_Diffuse_Tilted | 1,2 | 1 | 0.0446 | 4.23e-10 | negative |
| 5 | Weather_R | 12 | 12 | 0.0252 | 0.00112 | positive |
| 5 | Weather_T | 1,12 | 1 | 0.0224 | 0.00511 | negative |
| 6 | Active_Pow | 1,2,3,4,5,8,9,12 | 1 | 0.6153 | 0 | positive |
| 6 | Radiation_Global_Tilted | 1,2,3,4,5,6 | 1 | 0.1035 | 5.5e-89 | negative |
| 6 | Radiation_Diffuse_Tilted | 1,2,3,5,7,9 | 1 | 0.0336 | 4.47e-10 | negative |
| 6 | cos_day_of_year | 1,2,3,4,5,6,7,8,9,10,11,12 | 3 | 0.0296 | 5.58e-08 | positive |
| 6 | Weather_T | 1 | 1 | 0.0160 | 0.00796 | negative |
| 7 | Active_Pow | 1,2,3 | 1 | 0.6424 | 0 | positive |
| 7 | Radiation_Global_Tilted | 1,2,3,4,7,8,11 | 1 | 0.1165 | 9.95e-103 | positive |
| 7 | solar_elevation | 1,2,3,4,5,6,7,8,9,10,11,12 | 9 | 0.0425 | 2.33e-14 | positive |
| 7 | Radiation_Diffuse_Tilted | 1,2,4,5,6,7,8,9,10,11,12 | 1 | 0.0400 | 7.25e-13 | positive |
| 7 | Weather_T | 1,2,3 | 2 | 0.0329 | 5.73e-09 | positive |
