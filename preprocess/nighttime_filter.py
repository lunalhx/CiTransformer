"""
nighttime_filter.py
-------------------
光伏数据夜间标注模块

功能：
    根据 Alice Springs 站点的经纬度，利用 pvlib 计算太阳高度角，
    在不删除任何原始行的前提下，为 DataFrame 添加:
        - solar_elevation  : 太阳高度角（单位：度）
        - day_night_label  : 0 = 夜晚（高度角 < 5°），1 = 白天（高度角 ≥ 5°）
        - 周期时间特征     : 日内 / 年内的 sin-cos 编码

依赖：
    pip install pvlib pandas numpy
"""

import numpy as np
import pandas as pd
import pvlib


# ==============================================================================
# 模块级常量与配置
# ==============================================================================

# ── 1. 站点常量（Alice Springs, Australia） ──
SITE_LAT      = -23.762          # 纬度（南纬为负）
SITE_LON      = 133.875          # 经度（东经为正）
SITE_ALT      = 546.0            # 海拔（m）
SITE_TZ       = "Australia/Darwin"  # 时区（UTC+9:30，无夏令时）
ELEVATION_THR = 5.0              # 白天判定阈值（太阳高度角，单位：度）

# ── 2. 特征组常量 ──
CORE_FEATURE_COLS = [
    "Active_Pow",                  # 有功功率（预测目标）
    "Radiation_Global_Tilted",     # 总倾斜辐照度
    "Radiation_Diffuse_Tilted",    # 散射倾斜辐照度
    "Weather_T",                   # 环境温度
    "Weather_R",                   # 相对湿度
    "solar_elevation",             # 太阳高度角
    "sin_time_of_day",             # 日内周期特征（sin）
    "cos_time_of_day",             # 日内周期特征（cos）
    "sin_day_of_year",             # 年内周期特征（sin）
    "cos_day_of_year",             # 年内周期特征（cos）
    "day_night_label",             # 昼夜物理态势标签
]

NIGHTTIME_ZERO_COLS = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

IRRADIANCE_COLS = [
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

TARGET_COL = "Active_Pow"
PASSTHROUGH_COL = "day_night_label"
PV_GAP_RULE_COLS = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]
WEATHER_GAP_RULE_COLS = [
    "Weather_T",
    "Weather_R",
]
PV_SHORT_GAP_MAX = 3
PV_MEDIUM_GAP_MAX = 12
WEATHER_GAP_MAX = 12
CYCLICAL_TIME_FEATURE_COLS = [
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
]



# ── 3. 原始 CSV 列名映射（raw → 模块内部列名）──
RAW_COL_MAP = {
    "Active_Power"                   : "Active_Pow",
    "Weather_Temperature_Celsius"    : "Weather_T",
    "Weather_Relative_Humidity"      : "Weather_R",
    # 以下列名在原始 CSV 中已与代码一致，无需重命名：
    # "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted"
}


# ==============================================================================
# 第一部分：核心特征生成与提取
# ==============================================================================

def _ensure_site_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DatetimeIndex 使用站点本地时区。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"df.index 必须是 pd.DatetimeIndex，当前类型为 {type(df.index).__name__}。")

    if df.index.tz is None:
        print(f"[INFO] DatetimeIndex 无时区信息，本地化为 '{SITE_TZ}'。")
        df = df.copy()
        df.index = df.index.tz_localize(SITE_TZ)
    elif str(df.index.tz) != SITE_TZ:
        print(f"[INFO] DatetimeIndex 时区为 '{df.index.tz}'，转换为 '{SITE_TZ}'。")
        df = df.copy()
        df.index = df.index.tz_convert(SITE_TZ)
    else:
        df = df.copy()

    return df


def add_day_night_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    为光伏时间序列 DataFrame 添加太阳高度角列与昼夜标签列。
    """
    df = _ensure_site_timezone(df)

    solar_pos = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=SITE_LAT,
        longitude=SITE_LON,
        altitude=SITE_ALT,
    )

    df = df.assign(solar_elevation=solar_pos["elevation"].values)
    df["day_night_label"] = np.where(
        df["solar_elevation"] < ELEVATION_THR,
        0,
        1,
    )

    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于站点本地时间添加日内与年内周期特征。

    说明：
    - 仅使用本地时间，不转换为 UTC
    - 自动处理闰年（365 / 366 天）
    - 输出列统一为 float 类型
    """
    df = _ensure_site_timezone(df)
    local_index = df.index

    minutes_of_day = (
        local_index.hour.to_numpy(dtype="float64") * 60.0
        + local_index.minute.to_numpy(dtype="float64")
    )
    angle_day = 2.0 * np.pi * minutes_of_day / 1440.0

    day_of_year = local_index.dayofyear.to_numpy(dtype="float64")
    days_in_year = np.where(local_index.is_leap_year, 366.0, 365.0)
    year_progress = (day_of_year - 1.0 + minutes_of_day / 1440.0) / days_in_year
    angle_year = 2.0 * np.pi * year_progress

    df["sin_time_of_day"] = np.sin(angle_day).astype("float64")
    df["cos_time_of_day"] = np.cos(angle_day).astype("float64")
    df["sin_day_of_year"] = np.sin(angle_year).astype("float64")
    df["cos_day_of_year"] = np.cos(angle_year).astype("float64")

    return df


def select_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从包含全部列的 DataFrame 中筛选出核心特征列，保留原始 DatetimeIndex。
    """
    missing = [col for col in CORE_FEATURE_COLS if col not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame 中缺少以下必需列，请先完成前置处理步骤：\n"
            f"  缺失列: {missing}\n"
            f"  提示: solar_elevation 与 day_night_label 需先调用 add_day_night_labels(df) 生成；\n"
            f"        周期时间特征需先调用 add_cyclical_time_features(df) 生成。"
        )

    df_core = df[CORE_FEATURE_COLS].copy()
    return df_core


# ==============================================================================
# 第二部分：数据清洗
# ==============================================================================

def _build_nan_run_lengths(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    计算每个缺失点所属连续缺失段的长度。

    Returns
    -------
    row_run_lengths:
        与原始索引对齐；非缺失位置为 0，缺失位置为其所属缺失段长度。
    segment_lengths:
        每个连续缺失段的长度（每段一条）。
    """
    null_mask = series.isna()
    row_run_lengths = pd.Series(0, index=series.index, dtype="int64")

    if not null_mask.any():
        return row_run_lengths, pd.Series(dtype="int64")

    group_id = null_mask.ne(null_mask.shift(fill_value=False)).cumsum()
    segment_lengths = null_mask[null_mask].groupby(group_id[null_mask]).size().astype("int64")
    row_run_lengths.loc[null_mask] = segment_lengths.reindex(group_id[null_mask]).to_numpy()
    return row_run_lengths, segment_lengths


def _apply_gap_fill_rule(
    series: pd.Series,
    short_gap_max: int,
    medium_gap_max: int | None = None,
    fill_medium_gaps: bool = False,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """
    按连续缺失段长度执行分层处理。

    规则说明：
    - short gap: 允许严格因果前向填充
    - medium gap: 用于敏感性实验，默认不插值
    - long gap: 不插值，保留为无效段
    """
    series = series.copy()
    null_mask = series.isna()
    invalid_mask = pd.Series(False, index=series.index, dtype="bool")

    if not null_mask.any():
        return series, invalid_mask, {
            "missing_rows": 0,
            "filled_rows": 0,
            "invalid_rows": 0,
            "short_segments": 0,
            "medium_segments": 0,
            "long_segments": 0,
        }

    row_run_lengths, segment_lengths = _build_nan_run_lengths(series)
    causal_filled = series.ffill()

    short_mask = null_mask & row_run_lengths.le(short_gap_max)
    medium_mask = pd.Series(False, index=series.index, dtype="bool")
    fill_mask = short_mask.copy()

    if medium_gap_max is None:
        long_mask = null_mask & row_run_lengths.gt(short_gap_max)
        medium_segments = 0
        long_segments = int(segment_lengths.gt(short_gap_max).sum())
    else:
        medium_mask = null_mask & row_run_lengths.gt(short_gap_max) & row_run_lengths.le(medium_gap_max)
        long_mask = null_mask & row_run_lengths.gt(medium_gap_max)
        if fill_medium_gaps:
            fill_mask = fill_mask | medium_mask
        medium_segments = int(
            ((segment_lengths > short_gap_max) & (segment_lengths <= medium_gap_max)).sum()
        )
        long_segments = int((segment_lengths > medium_gap_max).sum())

    series.loc[fill_mask] = causal_filled.loc[fill_mask]

    unresolved_fill_mask = fill_mask & series.isna()
    invalid_mask = invalid_mask | unresolved_fill_mask | long_mask
    if medium_gap_max is not None and not fill_medium_gaps:
        invalid_mask = invalid_mask | medium_mask

    stats = {
        "missing_rows": int(null_mask.sum()),
        "filled_rows": int((fill_mask & series.notna()).sum()),
        "invalid_rows": int(invalid_mask.sum()),
        "short_segments": int(segment_lengths.le(short_gap_max).sum()),
        "medium_segments": medium_segments,
        "long_segments": long_segments,
    }
    return series, invalid_mask, stats


def clean_anomalies_causal_safe(
    df: pd.DataFrame,
    max_power_capacity: float = 11.55,
    max_irradiance: float = 1300.0,
    fill_medium_pv_gaps: bool = True,
) -> pd.DataFrame:
    """
    对核心特征 DataFrame 进行因果安全的异常与断点清洗。

    缺失处理规则：
    - Active_Pow / Radiation_*: 连续缺失 <= 3 点时严格因果前向填充；
      4~12 点默认执行严格因果前向填充；> 12 点视为无效段。
    - Weather_T / Weather_R: 连续缺失 <= 12 点时严格因果前向填充；
      > 12 点视为无效段。

    说明：
    - 本函数默认对 4~12 点的功率/辐照度缺失段执行严格因果前向填充；
      如需做保守敏感性实验，可传入 fill_medium_pv_gaps=False。
    - 所有填补仅使用过去已观测值，不使用未来信息。
    - 所有未被允许填补的缺失段都会在最终输出前裁剪掉。
    """
    df = df.copy()
    df.attrs["max_power_capacity"] = max_power_capacity

    # 3.1 夜间底噪绝对清零
    night_mask = df["day_night_label"] == 0
    df.loc[night_mask, NIGHTTIME_ZERO_COLS] = 0.0
    zeroed_count = night_mask.sum()
    print(f"[INFO] 步骤 3.1：已将 {zeroed_count} 个夜间时间步的 {NIGHTTIME_ZERO_COLS} 强制清零。")

    # 3.2 分层缺失处理
    invalid_row_mask = pd.Series(False, index=df.index, dtype="bool")

    for col in PV_GAP_RULE_COLS:
        df[col], col_invalid_mask, stats = _apply_gap_fill_rule(
            df[col],
            short_gap_max=PV_SHORT_GAP_MAX,
            medium_gap_max=PV_MEDIUM_GAP_MAX,
            fill_medium_gaps=fill_medium_pv_gaps,
        )
        invalid_row_mask = invalid_row_mask | col_invalid_mask
        print(
            f"[INFO] 步骤 3.2：{col} 缺失处理完成，"
            f"短缺失段={stats['short_segments']}，"
            f"中等缺失段={stats['medium_segments']}，"
            f"长缺失段={stats['long_segments']}，"
            f"已填补点数={stats['filled_rows']}，"
            f"无效点数={stats['invalid_rows']}。"
        )

    for col in WEATHER_GAP_RULE_COLS:
        df[col], col_invalid_mask, stats = _apply_gap_fill_rule(
            df[col],
            short_gap_max=WEATHER_GAP_MAX,
        )
        invalid_row_mask = invalid_row_mask | col_invalid_mask
        print(
            f"[INFO] 步骤 3.2：{col} 缺失处理完成，"
            f"可填补缺失段={stats['short_segments']}，"
            f"长缺失段={stats['long_segments']}，"
            f"已填补点数={stats['filled_rows']}，"
            f"无效点数={stats['invalid_rows']}。"
        )

    remaining_nan_mask = df[CORE_FEATURE_COLS].isna().any(axis=1)
    invalid_row_mask = invalid_row_mask | remaining_nan_mask

    rows_before = len(df)
    df = df.loc[~invalid_row_mask].copy()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"[INFO] 步骤 3.2：共有 {rows_dropped} 行因落入无效缺失段而被裁剪。")

    # 3.3 物理边界截断
    df["Active_Pow"] = df["Active_Pow"].clip(lower=0.0, upper=max_power_capacity)
    for col in IRRADIANCE_COLS:
        df[col] = df[col].clip(lower=0.0, upper=max_irradiance)
        
    print(f"[INFO] 步骤 3.3：Active_Pow 截断至 [0, {max_power_capacity}] kW；辐照度截断至 [0, {max_irradiance}] W/m²。")

    return df


# ==============================================================================
# 第三部分：真实数据文件处理
# ==============================================================================

def process_raw_file(
    raw_csv_path: str,
    output_dir: str,
    timestamp_col: str = "timestamp",
) -> None:
    """
    读取原始 CSV 文件，执行完整预处理流水线，并将结果写入 output_dir。

    输出文件（均含 DatetimeIndex）：
        - nighttime_labeled.csv   : 添加 solar_elevation / day_night_label / 周期时间特征后的全量数据
        - core_features_clean.csv : 按分层缺失规则裁剪后的核心特征（11 列）
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] 读取原始文件：{raw_csv_path}")
    df_raw = pd.read_csv(
        raw_csv_path,
        index_col=timestamp_col,
        parse_dates=True,
    )
    print(f"[INFO] 原始数据形状：{df_raw.shape}，时间范围：{df_raw.index.min()} ~ {df_raw.index.max()}")

    # 步骤 1：列名规范化
    df_raw = df_raw.rename(columns=RAW_COL_MAP)
    print(f"[INFO] 列名映射完成，当前列：{df_raw.columns.tolist()}")

    # 步骤 2：添加昼夜标签
    df_labeled = add_day_night_labels(df_raw)

    # 步骤 3：添加周期时间特征
    df_labeled = add_cyclical_time_features(df_labeled)
    out_labeled = os.path.join(output_dir, "nighttime_labeled.csv")
    df_labeled.to_csv(out_labeled)
    print(f"[INFO] 已保存 → {out_labeled}  shape={df_labeled.shape}")

    # 步骤 4：提取核心特征
    df_core = select_core_features(df_labeled)

    # 步骤 5：数据清洗
    df_clean = clean_anomalies_causal_safe(df_core)
    out_clean = os.path.join(output_dir, "core_features_clean.csv")
    df_clean.to_csv(out_clean)
    print(f"[INFO] 已保存 → {out_clean}  shape={df_clean.shape}")

    print("\n✅ 真实数据处理完成，所有输出文件已写入：", output_dir)


# ==============================================================================
# 单元测试与快速验证
# ==============================================================================
if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("🚀 [开始全模块流水线测试]")
    print("=" * 60)

    # ── 1. 构造模拟数据 ──
    date_rng = pd.date_range("2024-01-15 00:00:00", "2024-01-16 23:30:00", freq="30min", tz=SITE_TZ)
    rng = np.random.default_rng(seed=42)
    df_raw = pd.DataFrame({"power_kw": rng.uniform(0, 500, len(date_rng))}, index=date_rng)
    
    # ── 2. 测试 add_day_night_labels / add_cyclical_time_features ──
    df_labeled = add_day_night_labels(df_raw)
    df_labeled = add_cyclical_time_features(df_labeled)
    assert len(df_labeled) == len(df_raw), "数据长度断言失败"
    assert "solar_elevation" in df_labeled.columns and "day_night_label" in df_labeled.columns
    assert set(CYCLICAL_TIME_FEATURE_COLS).issubset(df_labeled.columns), "周期时间特征缺失"
    assert all(pd.api.types.is_float_dtype(df_labeled[col]) for col in CYCLICAL_TIME_FEATURE_COLS), "周期时间特征必须为 float"
    assert df_labeled[CYCLICAL_TIME_FEATURE_COLS].isna().sum().sum() == 0, "周期时间特征不应出现 NaN"
    print("✅ add_day_night_labels / add_cyclical_time_features 测试通过")
    
    # 构建后续流程使用的多列特征
    df_labeled["Radiation_Global_Tilted"]  = rng.uniform(0, 1000, len(df_labeled))
    df_labeled["Radiation_Diffuse_Tilted"] = rng.uniform(0, 300,  len(df_labeled))
    df_labeled["Weather_T"]               = rng.uniform(15, 45,   len(df_labeled))
    df_labeled["Weather_R"]               = rng.uniform(10, 90,   len(df_labeled))
    df_labeled["Active_Pow"]              = rng.uniform(0, 500,   len(df_labeled))
    
    # ── 3. 测试 select_core_features ──
    df_core = select_core_features(df_labeled)
    assert list(df_core.columns) == CORE_FEATURE_COLS
    assert len(df_core) == len(df_labeled)
    print("✅ select_core_features 测试通过")
    
    # ── 4. 测试 clean_anomalies_causal_safe ──
    df_dirty = df_core.copy()
    # 仅在白天样本中注入不同长度的缺失段，验证分层规则是否生效
    day_index = df_dirty.index[df_dirty["day_night_label"] == 1]
    short_gap_idx = day_index[3:6]
    medium_gap_idx = day_index[8:13]
    weather_fill_idx = day_index[14:26]
    weather_invalid_idx = day_index[26:39]

    df_dirty.loc[short_gap_idx, "Active_Pow"] = np.nan
    df_dirty.loc[medium_gap_idx, "Radiation_Global_Tilted"] = np.nan
    df_dirty.loc[weather_fill_idx, "Weather_T"] = np.nan
    df_dirty.loc[weather_invalid_idx, "Weather_R"] = np.nan
    df_dirty.iloc[10, df_dirty.columns.get_loc("Active_Pow")] = 999.0
    night_rows = df_dirty[df_dirty["day_night_label"] == 0].index[:3]
    df_dirty.loc[night_rows, "Active_Pow"] = 0.003
    
    df_clean = clean_anomalies_causal_safe(df_dirty)
    assert df_clean.isna().sum().sum() == 0, "NaN 断言失败"
    assert df_clean["Active_Pow"].max() <= 11.55, "上限断言失败"
    assert df_clean["Active_Pow"].min() >= 0.0, "下限断言失败"
    assert short_gap_idx.isin(df_clean.index).all(), "短缺失段应被保留并以前向填充方式处理"
    assert medium_gap_idx.isin(df_clean.index).all(), "中等功率/辐照度缺失段默认应被前向填充保留"
    assert weather_fill_idx.isin(df_clean.index).all(), "12 点以内气象缺失段应被前向填充保留"
    assert (~weather_invalid_idx.isin(df_clean.index)).all(), "超过 12 点的气象缺失段应裁剪"
    night_pow = df_clean.loc[df_clean["day_night_label"] == 0, "Active_Pow"]
    assert night_pow.max() == 0.0, "夜间清零断言失败"
    for col in CYCLICAL_TIME_FEATURE_COLS:
        assert df_clean[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all(), f"{col} 超出 [-1, 1] 范围"
    print("✅ clean_anomalies_causal_safe 测试通过")
    
    print("\n" + "=" * 60)
    print("🎉 [全流水线测试圆满完成]")
    print("=" * 60)

    # ── 7. 处理真实数据文件 ──
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    _raw_csv = os.path.join(_project_root, "data", "raw", "91-Site_DKA-M9_B-Phase.csv")
    _out_dir = os.path.join(_project_root, "data", "processed")

    if os.path.exists(_raw_csv):
        print("\n" + "=" * 60)
        print("📂 [开始处理真实数据文件]")
        print("=" * 60)
        process_raw_file(
            raw_csv_path=_raw_csv,
            output_dir=_out_dir,
        )
    else:
        print(f"\n[WARN] 真实数据文件不存在，跳过：{_raw_csv}")
