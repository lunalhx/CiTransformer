# Raw PV Data Audit Report

- Input file: `/Users/lunalhx/Desktop/202509/CiTransformer/data/raw/91-Site_DKA-M9_B-Phase.csv`
- Detected time column: `timestamp`
- Time span: 2013-08-14 15:35:00 to 2025-08-23 05:20:00
- Rows after timestamp parsing: 1250668
- Expected sampling interval: 0 days 00:05:00
- Expected-delta ratio: 99.9747%
- Larger-than-expected timestamp gaps: 317
- Estimated missing timestamp steps: 14106

## Missing Rate by Column
- `Wind_Speed`: 73.201% (915501 / 1250668)
- `Radiation_Global_Tilted`: 11.734% (146755 / 1250668)
- `Radiation_Diffuse_Tilted`: 8.366% (104628 / 1250668)
- `Weather_Temperature_Celsius`: 2.524% (31566 / 1250668)
- `Weather_Relative_Humidity`: 2.523% (31557 / 1250668)
- `Diffuse_Horizontal_Radiation`: 2.520% (31520 / 1250668)
- `Global_Horizontal_Radiation`: 2.520% (31519 / 1250668)
- `Wind_Direction`: 2.520% (31514 / 1250668)
- `Weather_Daily_Rainfall`: 2.520% (31511 / 1250668)
- `Active_Energy_Delivered_Received`: 2.149% (26871 / 1250668)
- `Current_Phase_Average`: 2.144% (26818 / 1250668)
- `Active_Power`: 2.144% (26818 / 1250668)

## Long Missing Runs
- `Wind_Speed`: max run 915480 rows, long runs 2, segments 3
- `Radiation_Global_Tilted`: max run 43435 rows, long runs 58, segments 15197
- `Radiation_Diffuse_Tilted`: max run 13655 rows, long runs 58, segments 15208
- `Diffuse_Horizontal_Radiation`: max run 6057 rows, long runs 4, segments 14005
- `Global_Horizontal_Radiation`: max run 6057 rows, long runs 4, segments 14004
- `Weather_Daily_Rainfall`: max run 6057 rows, long runs 4, segments 14001
- `Weather_Relative_Humidity`: max run 6057 rows, long runs 4, segments 14030
- `Weather_Temperature_Celsius`: max run 6057 rows, long runs 4, segments 14037
- `Wind_Direction`: max run 6057 rows, long runs 4, segments 14002
- `Active_Energy_Delivered_Received`: max run 862 rows, long runs 3, segments 14799
- `Active_Power`: max run 862 rows, long runs 3, segments 14766
- `Current_Phase_Average`: max run 862 rows, long runs 3, segments 14766

## Wind Variables
- Wind-speed columns: ['Wind_Speed']
- Wind-related columns: ['Wind_Speed', 'Wind_Direction']

| year | rows | wind missing % | wind complete rows % |
| --- | ---: | ---: | ---: |
| 2013 | 40127 | 0.052 | 99.948 |
| 2014 | 105111 | 0.000 | 100.000 |
| 2015 | 105120 | 0.000 | 100.000 |
| 2016 | 105408 | 19.522 | 80.478 |
| 2017 | 104614 | 100.000 | 0.000 |
| 2018 | 105120 | 100.000 | 0.000 |
| 2019 | 102905 | 100.000 | 0.000 |
| 2020 | 105381 | 100.000 | 0.000 |
| 2021 | 100540 | 100.000 | 0.000 |
| 2022 | 102901 | 100.000 | 0.000 |
| 2023 | 101797 | 100.000 | 0.000 |
| 2024 | 104245 | 100.000 | 0.000 |
| 2025 | 67399 | 100.000 | 0.000 |

## Candidate Modeling Intervals
| days | start | end exclusive | rows | score | required complete % | wind complete % | time coverage % | expected delta % |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 545 | 2015-04-06 00:00:00 | 2016-10-02 00:00:00 | 156960 | 99.683 | 99.094 | 100.000 | 100.000 | 100.000 |
| 545 | 2015-03-07 00:00:00 | 2016-09-02 00:00:00 | 156960 | 99.682 | 99.092 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-10-03 00:00:00 | 2016-10-02 00:00:00 | 105120 | 99.664 | 99.039 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-03-07 00:00:00 | 2016-03-06 00:00:00 | 105120 | 99.654 | 99.012 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-08-04 00:00:00 | 2016-08-03 00:00:00 | 105120 | 99.590 | 98.830 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-09-03 00:00:00 | 2016-09-02 00:00:00 | 105120 | 99.590 | 98.830 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-06-05 00:00:00 | 2016-06-04 00:00:00 | 105120 | 99.586 | 98.817 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-07-05 00:00:00 | 2016-07-04 00:00:00 | 105120 | 99.585 | 98.816 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-04-06 00:00:00 | 2016-04-05 00:00:00 | 105120 | 99.561 | 98.746 | 100.000 | 100.000 | 100.000 |
| 365 | 2015-05-06 00:00:00 | 2016-05-05 00:00:00 | 105120 | 99.532 | 98.663 | 100.000 | 100.000 | 100.000 |
| 545 | 2015-05-06 00:00:00 | 2016-11-01 00:00:00 | 156960 | 99.204 | 99.097 | 98.082 | 100.000 | 100.000 |
| 365 | 2015-11-02 00:00:00 | 2016-11-01 00:00:00 | 105120 | 98.947 | 99.037 | 97.137 | 100.000 | 100.000 |
| 365 | 2013-09-13 00:00:00 | 2014-09-13 00:00:00 | 105105 | 98.780 | 96.540 | 99.980 | 99.986 | 99.990 |
| 545 | 2015-02-05 00:00:00 | 2016-08-03 00:00:00 | 156960 | 98.226 | 94.932 | 100.000 | 100.000 | 100.000 |
| 545 | 2015-06-05 00:00:00 | 2016-12-01 00:00:00 | 156960 | 97.861 | 99.192 | 92.578 | 100.000 | 100.000 |
| 365 | 2013-08-14 00:00:00 | 2014-08-14 00:00:00 | 104918 | 97.831 | 93.877 | 99.980 | 99.808 | 99.990 |
| 365 | 2015-02-05 00:00:00 | 2016-02-05 00:00:00 | 105120 | 97.531 | 92.947 | 100.000 | 100.000 | 100.000 |
| 365 | 2013-10-13 00:00:00 | 2014-10-13 00:00:00 | 105105 | 97.494 | 92.864 | 99.980 | 99.986 | 99.990 |
| 365 | 2015-12-02 00:00:00 | 2016-12-01 00:00:00 | 105120 | 96.891 | 99.034 | 88.917 | 100.000 | 100.000 |
| 545 | 2015-07-05 00:00:00 | 2016-12-31 00:00:00 | 156960 | 96.404 | 98.960 | 87.073 | 100.000 | 100.000 |

Interpretation note: candidate scores prioritize required PV/weather completeness, wind-speed availability, timestamp continuity, and seasonal coverage. They are intended to shortlist intervals for modeling, not to replace research-design judgment.
