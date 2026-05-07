"""
轻量级机器学习分类器 - 基于技术指标预测市场阶段

使用 sklearn RandomForestClassifier，利用回测历史数据训练。
训练数据不足时自动降级到规则系统。

工作流程：
1. 从回测数据生成训练样本（特征 = 技术指标, 标签 = 未来N周收益是否>阈值）
2. 训练随机森林分类器
3. 对新数据预测阶段概率
4. 与规则系统结果加权融合
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# sklearn 可选导入
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLStageClassifier:
    """
    ML阶段分类器

    特征:
    - diff30_pct: 价格vs30周均线偏离
    - diff10_pct: 价格vs10周均线偏离
    - ma30_slope: 30周均线斜率
    - ma30_r2: 均线拟合度
    - rsi: RSI值
    - ret_4w: 4周涨幅
    - ret_8w: 8周涨幅
    - ma_arrangement: 均线排列
    - vol_ratio: 成交量比率
    """

    FEATURE_COLS = [
        "diff30_pct", "diff10_pct", "ma30_slope", "ma30_r2",
        "rsi", "ret_4w", "ret_8w", "ma_arrangement", "vol_ratio"
    ]

    STAGE_MAP = {0: "unknown", 1: "accumulation", 2: "rising", 3: "top", 4: "falling"}

    def __init__(self, min_samples: int = 100):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.min_samples = min_samples
        self.is_trained = False
        self.feature_importance: Dict[str, float] = {}

    @property
    def available(self) -> bool:
        return SKLEARN_AVAILABLE and self.is_trained

    def _prepare_features(self, metrics_list: list) -> np.ndarray:
        """从指标列表构建特征矩阵"""
        rows = []
        for m in metrics_list:
            row = [m.get(col, 0) for col in self.FEATURE_COLS]
            rows.append(row)
        X = np.array(rows)
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def _generate_labels(self, future_returns: list, threshold: float = 0.02) -> np.ndarray:
        """
        根据未来收益生成阶段标签
        1: 筑底(收益>=0但<阈值) | 2: 上升(收益>=阈值)
        3: 顶部(高收益后回落) | 4: 下跌(收益<0)
        """
        labels = []
        for i in range(len(future_returns) - 1):
            current_ret = future_returns[i]
            next_ret = future_returns[i + 1] if i < len(future_returns) - 1 else 0

            if next_ret >= threshold:
                labels.append(2)  # 上升
            elif next_ret < -0.01:
                labels.append(4)  # 下跌
            elif current_ret > threshold and next_ret < current_ret * 0.5:
                labels.append(3)  # 顶部
            else:
                labels.append(1)  # 筑底
        return np.array(labels)

    def train_from_backtest(self, backtest_signals: pd.DataFrame) -> dict:
        """
        从回测信号数据训练模型

        Args:
            backtest_signals: DataFrame，包含指标列和未来收益率列
                            必须包含 FEATURE_COLS + 未来收益列
        """
        if not SKLEARN_AVAILABLE:
            return {"status": "sklean_not_available", "samples": 0}

        if len(backtest_signals) < self.min_samples:
            return {"status": "insufficient_data", "samples": len(backtest_signals)}

        # 构建特征
        X_list = []
        y_list = []
        for _, row in backtest_signals.iterrows():
            features = {}
            for col in self.FEATURE_COLS:
                val = row.get(col, 0)
                features[col] = val if not pd.isna(val) else 0
            X_list.append(features)

            # 用未来4周收益作为标签依据
            fwd_ret = row.get("8周", 0)  # 使用8周前向收益
            if pd.isna(fwd_ret):
                continue
            if fwd_ret > 3:
                stage = 2
            elif fwd_ret > 0:
                stage = 1
            elif fwd_ret > -3:
                stage = 4
            else:
                stage = 4
            y_list.append(stage)

        if len(X_list) < self.min_samples:
            return {"status": "insufficient_samples", "samples": len(X_list)}

        X = np.array([[v for v in x.values()] for x in X_list])
        y = np.array(y_list)

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练随机森林
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        # 特征重要性
        self.feature_importance = dict(zip(self.FEATURE_COLS,
                                           [round(v, 3) for v in self.model.feature_importances_]))

        # 评估
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))

        self.is_trained = True
        return {
            "status": "trained",
            "samples": len(X_list),
            "train_accuracy": round(train_acc, 3),
            "test_accuracy": round(test_acc, 3),
            "feature_importance": self.feature_importance
        }

    def predict_stage(self, metrics: dict) -> dict:
        """
        对单一样本预测阶段

        Args:
            metrics: 技术指标字典，包含 FEATURE_COLS 中的字段

        Returns:
            各阶段概率和融合后的阶段判断
        """
        default = {
            "ml_stage": 0, "ml_confidence": 0.0,
            "probs": {"accumulation": 0.0, "rising": 0.0, "top": 0.0, "falling": 0.0},
            "available": False
        }

        if not self.available:
            return default

        try:
            features = np.array([[metrics.get(col, 0) for col in self.FEATURE_COLS]])
            features_scaled = self.scaler.transform(features)

            probs = self.model.predict_proba(features_scaled)[0]

            # 模型可能不会返回所有4个类别的概率
            stage_probs = {"accumulation": 0.0, "rising": 0.0, "top": 0.0, "falling": 0.0}
            stage_names = {1: "accumulation", 2: "rising", 3: "top", 4: "falling"}

            for i, cls in enumerate(self.model.classes_):
                name = stage_names.get(int(cls), "unknown")
                if name in stage_probs:
                    stage_probs[name] = round(float(probs[i]), 3)

            ml_stage = int(self.model.predict(features_scaled)[0])
            ml_confidence = float(max(probs))

            return {
                "ml_stage": ml_stage,
                "ml_confidence": round(ml_confidence, 3),
                "probs": stage_probs,
                "available": True
            }
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return default

    def fuse_with_rules(self, fuzzy_result: dict, ml_result: dict,
                        ml_weight: float = 0.3) -> dict:
        """
        融合ML和规则系统的阶段判断

        当ML不可用时，完全使用规则系统结果
        """
        if not ml_result.get("available", False):
            return fuzzy_result

        # 概率加权融合
        fuzzy_probs = fuzzy_result.get("stage_probs", {})
        ml_probs = ml_result.get("probs", {})

        fused_probs = {}
        for stage_name in ["accumulation", "rising", "top", "falling"]:
            fp = fuzzy_probs.get(stage_name, 0)
            mp = ml_probs.get(stage_name, 0)
            fused_probs[stage_name] = round(fp * (1 - ml_weight) + mp * ml_weight, 3)

        # 取最高概率
        stage_map = {"accumulation": 1, "rising": 2, "top": 3, "falling": 4}
        max_stage = max(fused_probs, key=lambda s: fused_probs[s])
        stage = stage_map[max_stage]
        confidence = fused_probs[max_stage]

        return {
            "stage": stage,
            "confidence": round(min(confidence, 1.0), 3),
            "reason": fuzzy_result.get("reason", "") + f" (ML权重{ml_weight:.0%})",
            "stage_probs": fused_probs,
            "key_metrics": fuzzy_result.get("key_metrics", {}),
            "ml_info": {
                "ml_stage": ml_result.get("ml_stage"),
                "ml_confidence": ml_result.get("ml_confidence"),
                "feature_importance": self.feature_importance if self.feature_importance else {}
            }
        }

    def get_training_status(self) -> dict:
        """获取模型训练状态"""
        if not SKLEARN_AVAILABLE:
            return {"available": False, "reason": "sklearn 未安装"}
        if not self.is_trained:
            return {"available": False, "reason": "未训练"}
        return {
            "available": True,
            "trained": True,
            "feature_importance": self.feature_importance
        }
