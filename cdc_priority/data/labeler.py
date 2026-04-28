import pandas as pd


# 默认标签规则采用“一级指标 + 二级特征”的层次化结构，
# 这样既便于论文表述，也便于后续做权重敏感性分析。
DEFAULT_LABELING_CONFIG = {
    "priority_score": {
        "components": {
            "business_importance": {
                "weight": 0.40,
                "numeric_weights": {
                    "business_value": 0.75,
                },
                "categorical_weights": {
                    "business_domain": 0.15,
                    "user_level": 0.10,
                },
                "invert_numeric": [],
                "hot_values": {
                    "business_domain": ["PAYMENT", "ORDER", "RISK", "INVENTORY"],
                    "user_level": ["VIP", "INTERNAL"],
                },
            },
            "timeliness": {
                "weight": 0.30,
                "numeric_weights": {
                    "queue_wait_time": 0.45,
                    "deadline_gap": 0.35,
                    "retry_count": 0.10,
                    "is_peak_hour": 0.10,
                },
                "categorical_weights": {},
                "invert_numeric": ["deadline_gap"],
                "hot_values": {},
            },
            "dependency_impact": {
                "weight": 0.20,
                "numeric_weights": {
                    "dependency_count": 0.60,
                    "consistency_risk": 0.40,
                },
                "categorical_weights": {
                    "event_type": 1.0,
                },
                "invert_numeric": [],
                "hot_values": {
                    "event_type": ["DELETE", "PAYMENT", "ALERT"],
                },
            },
            "execution_feasibility": {
                "weight": 0.10,
                "numeric_weights": {
                    "estimated_sync_cost": 0.70,
                    "source_load": 0.15,
                    "db_load": 0.15,
                },
                "categorical_weights": {},
                "invert_numeric": ["estimated_sync_cost", "source_load", "db_load"],
                "hot_values": {},
            },
        },
        "thresholds": {
            "medium": 0.40,
            "high": 0.70,
        },
    }
}


def _normalized_flag(frame: pd.DataFrame, column: str, hot_values: set[str]) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = frame[column].astype(str).str.upper()
    return values.isin(hot_values).astype("float32")


def _normalized_numeric(frame: pd.DataFrame, column: str, invert: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float32")
    max_value = float(values.max())
    min_value = float(values.min())
    if max_value - min_value <= 1e-9:
        normalized = pd.Series(0.0, index=frame.index, dtype="float32")
    else:
        normalized = (values - min_value) / (max_value - min_value)
    return 1.0 - normalized if invert else normalized


def _merged_labeling_config(labeling_config: dict | None) -> dict:
    merged = {
        "priority_score": {
            "components": {},
            "thresholds": dict(DEFAULT_LABELING_CONFIG["priority_score"]["thresholds"]),
        }
    }
    for component_name, component in DEFAULT_LABELING_CONFIG["priority_score"]["components"].items():
        merged["priority_score"]["components"][component_name] = {
            "weight": float(component["weight"]),
            "numeric_weights": dict(component["numeric_weights"]),
            "categorical_weights": dict(component["categorical_weights"]),
            "invert_numeric": list(component["invert_numeric"]),
            "hot_values": {
                key: list(values)
                for key, values in component["hot_values"].items()
            },
        }

    if not labeling_config:
        return merged

    priority_score = labeling_config.get("priority_score", {})
    if "thresholds" in priority_score:
        merged["priority_score"]["thresholds"].update(priority_score["thresholds"])

    components = priority_score.get("components")
    if components:
        for component_name, component in components.items():
            target = merged["priority_score"]["components"].setdefault(
                component_name,
                {
                    "weight": 0.0,
                    "numeric_weights": {},
                    "categorical_weights": {},
                    "invert_numeric": [],
                    "hot_values": {},
                },
            )
            if "weight" in component:
                target["weight"] = float(component["weight"])
            for key in ("numeric_weights", "categorical_weights"):
                target[key].update(component.get(key, {}))
            if "invert_numeric" in component:
                target["invert_numeric"] = list(component["invert_numeric"])
            if "hot_values" in component:
                for column, values in component["hot_values"].items():
                    target["hot_values"][column] = list(values)
        return merged

    # 兼容旧版“扁平字段加权”配置，避免历史配置文件直接失效。
    legacy_component = merged["priority_score"]["components"].setdefault(
        "legacy_priority",
        {
            "weight": 1.0,
            "numeric_weights": {},
            "categorical_weights": {},
            "invert_numeric": [],
            "hot_values": {},
        },
    )
    legacy_component["numeric_weights"].update(priority_score.get("numeric_weights", {}))
    legacy_component["categorical_weights"].update(priority_score.get("categorical_weights", {}))
    if "invert_numeric" in priority_score:
        legacy_component["invert_numeric"] = list(priority_score["invert_numeric"])
    if "hot_values" in priority_score:
        for column, values in priority_score["hot_values"].items():
            legacy_component["hot_values"][column] = list(values)
    if "weight" in priority_score:
        legacy_component["weight"] = float(priority_score["weight"])
    return merged


def _build_component_score(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype="float32")
    total_weight = 0.0
    invert_numeric = set(component.get("invert_numeric", []))

    for column, weight in component.get("numeric_weights", {}).items():
        weight_value = float(weight)
        score += weight_value * _normalized_numeric(
            frame,
            column,
            invert=column in invert_numeric,
        )
        total_weight += weight_value

    for column, weight in component.get("categorical_weights", {}).items():
        weight_value = float(weight)
        hot_values = set(str(value).upper() for value in component.get("hot_values", {}).get(column, []))
        score += weight_value * _normalized_flag(frame, column, hot_values)
        total_weight += weight_value

    # 先在组件内部归一化，避免某个组件因为子项较多而天然占更大权重。
    if total_weight <= 1e-9:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return score / total_weight


def build_priority_score(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.Series:
    config = _merged_labeling_config(labeling_config)["priority_score"]
    score = pd.Series(0.0, index=frame.index, dtype="float32")
    total_component_weight = 0.0

    # 最终优先级分数由多个一级指标加权汇总而成。
    for component in config["components"].values():
        component_weight = float(component.get("weight", 0.0))
        if component_weight <= 0:
            continue
        score += component_weight * _build_component_score(frame, component)
        total_component_weight += component_weight

    if total_component_weight <= 1e-9:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return score / total_component_weight


def attach_priority_label(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.DataFrame:
    labeled = frame.copy()
    if "priority_label" in labeled.columns:
        # 如果数据源已经带标签，则优先保留，避免重复覆盖人工或上游标签。
        labeled["priority_label"] = (
            labeled["priority_label"].fillna("medium").astype(str).str.lower()
        )
        return labeled

    config = _merged_labeling_config(labeling_config)["priority_score"]
    thresholds = config["thresholds"]
    priority_score = build_priority_score(labeled, labeling_config=labeling_config)
    labeled["priority_score"] = priority_score
    # 采用双阈值把连续分数映射成 high / medium / low 三分类标签。
    labeled["priority_label"] = "low"
    labeled.loc[priority_score >= float(thresholds["medium"]), "priority_label"] = "medium"
    labeled.loc[priority_score >= float(thresholds["high"]), "priority_label"] = "high"
    return labeled
