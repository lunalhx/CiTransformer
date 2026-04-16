"""
feature_scaling.py
------------------
光伏核心特征标准化模块。

功能：
    将标准化步骤从全局预处理中解耦，单独负责对清洗后的核心特征做缩放。

依赖：
    pip install pandas scikit-learn
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TARGET_COL = "Active_Pow"
EXOG_COLS = [
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_T",
    "Weather_R",
    "solar_elevation",
]
PASSTHROUGH_COL = "day_night_label"


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
        exo_scaler = StandardScaler()

        # 用物理边界哨兵 [0, max_power_capacity] 参与 fit，
        # 保证夜间的 Active_Pow=0 transform 后不会出现负值。
        power_cap = df.attrs.get("max_power_capacity", 11.55)
        sentinel = pd.DataFrame({TARGET_COL: [0.0, power_cap]})
        fit_data = pd.concat([df_day[[TARGET_COL]], sentinel], ignore_index=True)
        power_scaler.fit(fit_data)
        exo_scaler.fit(df_day[EXOG_COLS])

        print(
            f"[INFO] 特征缩放（训练）：基于 {day_mask.sum()} 个白天样本完成 fit。\n"
            f"  power_scaler data_range: [{power_scaler.data_min_[0]:.4f}, {power_scaler.data_max_[0]:.4f}]\n"
            f"  exo_scaler fitted."
        )

        scalers = {
            "power_scaler": power_scaler,
            "exo_scaler": exo_scaler,
        }
    else:
        power_scaler = scalers["power_scaler"]
        exo_scaler = scalers["exo_scaler"]
        print(f"[INFO] 特征缩放（推理）：使用外部 scalers 对 {len(df)} 行数据做 transform。")

    df[TARGET_COL] = power_scaler.transform(df[[TARGET_COL]])
    df[EXOG_COLS] = exo_scaler.transform(df[EXOG_COLS])

    if is_train:
        return df, scalers
    return df
