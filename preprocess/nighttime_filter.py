"""
nighttime_filter.py
-------------------
光伏数据夜间标注模块

功能：
    根据 Alice Springs 站点的经纬度，利用 pvlib 计算太阳高度角，
    在不删除任何原始行的前提下，为 DataFrame 添加:
        - solar_elevation  : 太阳高度角（单位：度）
        - day_night_label  : 0 = 夜晚（高度角 < 5°），1 = 白天（高度角 ≥ 5°）

依赖：
    pip install pvlib pandas numpy scikit-learn hmmlearn
"""

import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hmmlearn.hmm import GaussianHMM


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
    "Wind_Spee",                   # 风速
    "Weather_R",                   # 相对湿度
    "solar_elevation",             # 太阳高度角
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
EXOG_COLS = [
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_T",
    "Wind_Spee",
    "Weather_R",
    "solar_elevation",
]
PASSTHROUGH_COL = "day_night_label"

HMM_FEATURE_COLS = [TARGET_COL] + EXOG_COLS   # Active_Pow + 6 路气象/天文


# ==============================================================================
# 第一部分：核心特征生成与提取
# ==============================================================================

def add_day_night_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    为光伏时间序列 DataFrame 添加太阳高度角列与昼夜标签列。
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


def select_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从包含全部列的 DataFrame 中筛选出核心特征列，保留原始 DatetimeIndex。
    """
    missing = [col for col in CORE_FEATURE_COLS if col not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame 中缺少以下必需列，请先完成前置处理步骤：\n"
            f"  缺失列: {missing}\n"
            f"  提示: solar_elevation 与 day_night_label 需先调用 add_day_night_labels(df) 生成。"
        )

    df_core = df[CORE_FEATURE_COLS].copy()
    return df_core


# ==============================================================================
# 第二部分：数据清洗与标准化
# ==============================================================================

def clean_anomalies_causal_safe(
    df: pd.DataFrame,
    max_power_capacity: float = 11.55,
    max_irradiance: float = 1300.0,
) -> pd.DataFrame:
    """
    对核心特征 DataFrame 进行因果安全的异常与断点清洗。
    """
    df = df.copy()

    # 3.1 严格因果约束的缺失值修复（线性前向插值 + ffill兜底）
    df = df.interpolate(method="linear", limit_direction="forward")
    df = df.ffill()

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"[INFO] 步骤 3.1：头部 {rows_dropped} 行因含无法前向填充的 NaN 已被安全裁剪。")

    # 3.2 夜间底噪绝对清零
    night_mask = df["day_night_label"] == 0
    df.loc[night_mask, NIGHTTIME_ZERO_COLS] = 0.0
    zeroed_count = night_mask.sum()
    print(f"[INFO] 步骤 3.2：已将 {zeroed_count} 个夜间时间步的 {NIGHTTIME_ZERO_COLS} 强制清零。")

    # 3.3 物理边界截断
    df["Active_Pow"] = df["Active_Pow"].clip(lower=0.0, upper=max_power_capacity)
    for col in IRRADIANCE_COLS:
        df[col] = df[col].clip(lower=0.0, upper=max_irradiance)
        
    print(f"[INFO] 步骤 3.3：Active_Pow 截断至 [0, {max_power_capacity}] kW；辐照度截断至 [0, {max_irradiance}] W/m²。")

    return df


def scale_features(
    df: pd.DataFrame,
    is_train: bool = True,
    scalers: dict | None = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """
    对清洗后的核心特征 DataFrame 执行异构空间标准化。
    """
    if not is_train:
        if scalers is None:
            raise ValueError("is_train=False 时必须传入已拟合的 scalers 字典。")
        required_keys = {"power_scaler", "exo_scaler"}
        missing_keys = required_keys - set(scalers.keys())
        if missing_keys:
            raise ValueError(f"scalers 字典缺少键：{missing_keys}")

    df = df.copy()

    if is_train:
        day_mask = df[PASSTHROUGH_COL] == 1
        df_day = df.loc[day_mask]

        if df_day.empty:
            raise ValueError("传入的 DataFrame 中无白天样本，无法 fit Scaler。")

        power_scaler = MinMaxScaler(feature_range=(0, 1))
        exo_scaler   = StandardScaler()

        power_scaler.fit(df_day[[TARGET_COL]])
        exo_scaler.fit(df_day[EXOG_COLS])

        print(
            f"[INFO] 步骤 4（训练）：基于 {day_mask.sum()} 个白天样本完成 fit。\n"
            f"  power_scaler data_range: [{power_scaler.data_min_[0]:.4f}, {power_scaler.data_max_[0]:.4f}]\n"
            f"  exo_scaler fitted."
        )

        scalers = {
            "power_scaler": power_scaler,
            "exo_scaler"  : exo_scaler,
        }
    else:
        power_scaler = scalers["power_scaler"]
        exo_scaler   = scalers["exo_scaler"]
        print(f"[INFO] 步骤 4（推理）：使用外部 scalers 对 {len(df)} 行数据做 transform。")

    df[TARGET_COL]  = power_scaler.transform(df[[TARGET_COL]])
    df[EXOG_COLS]   = exo_scaler.transform(df[EXOG_COLS])

    if is_train:
        return df, scalers
    else:
        return df


# ==============================================================================
# 第三部分：解耦与特征重构
# ==============================================================================

def train_and_predict_hmm(
    df: pd.DataFrame,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    对标准化后的全量数据，提取白天序列训练 GMM-HMM，并重构 Regime 标签。
    """
    missing = [c for c in HMM_FEATURE_COLS + [PASSTHROUGH_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame 缺少以下必需列：{missing}")

    df = df.copy()
    df_daytime = df[df[PASSTHROUGH_COL] == 1].copy()

    if df_daytime.empty:
        raise ValueError("无白天记录，无法训练 HMM。")

    lengths_series = df_daytime.groupby(df_daytime.index.date, sort=True).size()
    lengths_array = lengths_series.values.tolist()

    print(f"[INFO] 步骤 5.1：共 {len(lengths_array)} 个自然日白天序列，总时间步 {sum(lengths_array)}。")

    X_train = df_daytime[HMM_FEATURE_COLS].values.astype(float)

    hmm_model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        verbose=False,
    )

    hmm_model.fit(X_train, lengths=lengths_array)
    raw_labels = hmm_model.predict(X_train, lengths=lengths_array)
    daytime_labels = raw_labels + 1

    print(f"[INFO] 步骤 5.2：HMM 训练完成（白天标签 {sorted(np.unique(daytime_labels))}）。")

    df["Regime"] = 0
    df.loc[df_daytime.index, "Regime"] = daytime_labels

    print(f"[INFO] 步骤 5.3：Regime 列已缝合完成（全量标签 {sorted(np.unique(df['Regime']))}）。")

    return df, hmm_model


# ==============================================================================
# 单元测试与快速验证
# ==============================================================================
if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("🚀 [开始全模块流水线测试]")
    print("=" * 60)

    # ── 1. 构造模拟数据 ──
    date_rng = pd.date_range("2024-01-15 00:00:00", "2024-01-15 23:30:00", freq="30min", tz=SITE_TZ)
    rng = np.random.default_rng(seed=42)
    df_raw = pd.DataFrame({"power_kw": rng.uniform(0, 500, len(date_rng))}, index=date_rng)
    
    # ── 2. 测试 add_day_night_labels ──
    df_labeled = add_day_night_labels(df_raw)
    assert len(df_labeled) == len(df_raw), "数据长度断言失败"
    assert "solar_elevation" in df_labeled.columns and "day_night_label" in df_labeled.columns
    print("✅ add_day_night_labels 测试通过")
    
    # 构建后续流程使用的多列特征
    df_labeled["Radiation_Global_Tilted"]  = rng.uniform(0, 1000, len(df_labeled))
    df_labeled["Radiation_Diffuse_Tilted"] = rng.uniform(0, 300,  len(df_labeled))
    df_labeled["Weather_T"]               = rng.uniform(15, 45,   len(df_labeled))
    df_labeled["Wind_Spee"]               = rng.uniform(0, 15,    len(df_labeled))
    df_labeled["Weather_R"]               = rng.uniform(10, 90,   len(df_labeled))
    df_labeled["Active_Pow"]              = rng.uniform(0, 500,   len(df_labeled))
    
    # ── 3. 测试 select_core_features ──
    df_core = select_core_features(df_labeled)
    assert list(df_core.columns) == CORE_FEATURE_COLS
    assert len(df_core) == len(df_labeled)
    print("✅ select_core_features 测试通过")
    
    # ── 4. 测试 clean_anomalies_causal_safe ──
    df_dirty = df_core.copy()
    # 注入异常进行测试
    df_dirty.iloc[3:6, df_dirty.columns.get_loc("Active_Pow")] = np.nan
    df_dirty.iloc[0:2, df_dirty.columns.get_loc("Weather_T")] = np.nan
    df_dirty.iloc[10, df_dirty.columns.get_loc("Active_Pow")] = 999.0
    night_rows = df_dirty[df_dirty["day_night_label"] == 0].index[:3]
    df_dirty.loc[night_rows, "Active_Pow"] = 0.003
    
    df_clean = clean_anomalies_causal_safe(df_dirty)
    assert df_clean.isna().sum().sum() == 0, "NaN 断言失败"
    assert df_clean["Active_Pow"].max() <= 11.55, "上限断言失败"
    assert df_clean["Active_Pow"].min() >= 0.0, "下限断言失败"
    night_pow = df_clean.loc[df_clean["day_night_label"] == 0, "Active_Pow"]
    assert night_pow.max() == 0.0, "夜间清零断言失败"
    print("✅ clean_anomalies_causal_safe 测试通过")
    
    # ── 5. 测试 scale_features ──
    df_scaled_train, fitted_scalers = scale_features(df_clean, is_train=True)
    df_scaled_infer = scale_features(df_clean, is_train=False, scalers=fitted_scalers)
    assert len(df_scaled_train) == len(df_clean)
    assert set(df_scaled_train[PASSTHROUGH_COL].unique()).issubset({0, 1})
    assert df_scaled_train["Active_Pow"].max() <= 1.0 + 1e-9
    assert df_scaled_train["Active_Pow"].min() >= 0.0 - 1e-9
    assert np.allclose(df_scaled_train[EXOG_COLS].values, df_scaled_infer[EXOG_COLS].values)
    print("✅ scale_features 测试通过")
    
    # ── 6. 测试 train_and_predict_hmm ──
    df_regime, trained_hmm = train_and_predict_hmm(df_scaled_train, n_components=3)
    assert len(df_regime) == len(df_scaled_train)
    assert "Regime" in df_regime.columns
    unique_regimes = set(df_regime["Regime"].unique())
    assert unique_regimes.issubset({0, 1, 2, 3})
    assert (df_regime.loc[df_regime[PASSTHROUGH_COL] == 0, "Regime"] == 0).all()
    assert (df_regime.loc[df_regime[PASSTHROUGH_COL] == 1, "Regime"] >= 1).all()
    assert hasattr(trained_hmm, "transmat_") and trained_hmm.transmat_.shape == (3, 3)
    print("✅ train_and_predict_hmm 测试通过")
    
    print("\n" + "=" * 60)
    print("🎉 [全流水线测试圆满完成]")
    print("=" * 60)
