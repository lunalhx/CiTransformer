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
| 2013 | 40127 | nan | nan |
| 2014 | 105111 | nan | nan |
| 2015 | 105120 | nan | nan |
| 2016 | 105408 | nan | nan |
| 2017 | 104614 | nan | nan |
| 2018 | 105120 | nan | nan |
| 2019 | 102905 | nan | nan |
| 2020 | 105381 | nan | nan |
| 2021 | 100540 | nan | nan |
| 2022 | 102901 | nan | nan |
| 2023 | 101797 | nan | nan |
| 2024 | 104245 | nan | nan |
| 2025 | 67399 | nan | nan |

## Candidate Modeling Intervals
| days | start | end exclusive | rows | score | required complete % | wind complete % | time coverage % | expected delta % |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 365 | 2016-06-29 00:00:00 | 2017-06-29 00:00:00 | 105103 | 99.775 | 99.628 | n/a | 99.984 | 99.999 |
| 365 | 2016-04-30 00:00:00 | 2017-04-30 00:00:00 | 105120 | 99.774 | 99.623 | n/a | 100.000 | 100.000 |
| 365 | 2016-05-30 00:00:00 | 2017-05-30 00:00:00 | 105103 | 99.772 | 99.623 | n/a | 99.984 | 99.999 |
| 365 | 2016-12-26 00:00:00 | 2017-12-26 00:00:00 | 104614 | 99.763 | 99.686 | n/a | 99.519 | 99.998 |
| 365 | 2016-10-27 00:00:00 | 2017-10-27 00:00:00 | 104614 | 99.672 | 99.534 | n/a | 99.519 | 99.998 |
| 365 | 2016-09-27 00:00:00 | 2017-09-27 00:00:00 | 104614 | 99.671 | 99.533 | n/a | 99.519 | 99.998 |
| 365 | 2016-07-29 00:00:00 | 2017-07-29 00:00:00 | 104614 | 99.670 | 99.531 | n/a | 99.519 | 99.998 |
| 365 | 2016-08-28 00:00:00 | 2017-08-28 00:00:00 | 104614 | 99.670 | 99.531 | n/a | 99.519 | 99.998 |
| 365 | 2016-11-26 00:00:00 | 2017-11-26 00:00:00 | 104614 | 99.638 | 99.477 | n/a | 99.519 | 99.998 |
| 365 | 2016-03-01 00:00:00 | 2017-03-01 00:00:00 | 105120 | 99.559 | 99.265 | n/a | 100.000 | 100.000 |
| 365 | 2016-03-31 00:00:00 | 2017-03-31 00:00:00 | 105120 | 99.558 | 99.264 | n/a | 100.000 | 100.000 |
| 730 | 2015-10-03 00:00:00 | 2017-10-02 00:00:00 | 209734 | 99.547 | 99.286 | n/a | 99.759 | 99.999 |
| 730 | 2015-11-02 00:00:00 | 2017-11-01 00:00:00 | 209734 | 99.547 | 99.285 | n/a | 99.759 | 99.999 |
| 730 | 2015-07-05 00:00:00 | 2017-07-04 00:00:00 | 210223 | 99.532 | 99.222 | n/a | 99.992 | 100.000 |
| 730 | 2015-06-05 00:00:00 | 2017-06-04 00:00:00 | 210223 | 99.531 | 99.220 | n/a | 99.992 | 100.000 |
| 730 | 2015-12-02 00:00:00 | 2017-12-01 00:00:00 | 209734 | 99.530 | 99.257 | n/a | 99.759 | 99.999 |
| 730 | 2015-05-06 00:00:00 | 2017-05-05 00:00:00 | 210240 | 99.486 | 99.144 | n/a | 100.000 | 100.000 |
| 730 | 2015-04-06 00:00:00 | 2017-04-05 00:00:00 | 210240 | 99.484 | 99.141 | n/a | 100.000 | 100.000 |
| 730 | 2015-09-03 00:00:00 | 2017-09-02 00:00:00 | 209734 | 99.483 | 99.179 | n/a | 99.759 | 99.999 |
| 730 | 2015-08-04 00:00:00 | 2017-08-03 00:00:00 | 209734 | 99.483 | 99.179 | n/a | 99.759 | 99.999 |

Interpretation note: candidate scores prioritize required PV/weather completeness, wind-speed availability, timestamp continuity, and seasonal coverage. They are intended to shortlist intervals for modeling, not to replace research-design judgment.
