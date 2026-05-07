"""
轻量级机器学习分类器 - 基于技术指标预测市场阶段

支持 sklearn RandomForestClassifier 和 XGBoost XGBClassifier。
训练数据不足时自动降级到规则系统。

工作流程：
1. 从回测数据生成训练样本（特征 = 技术指标, 标签 = 未来N周收益是否>阈值）
2. 训练分类器
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
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# xgboost 可选导入
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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

    def __init__(self, min_samples: int = 100, model_type: str = "xgb"):
        """
        Args:
            min_samples: 最少训练样本数
            model_type: 模型类型, "rf"=RandomForest, "xgb"=XGBoost (默认)
        """
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.min_samples = min_samples
        self.is_trained = False
        self.feature_importance: Dict[str, float] = {}
        self.model_type = model_type
        self._label_encoder: Optional[LabelEncoder] = None

    @property
    def available(self) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        if self.model_type == "xgb" and not XGBOOST_AVAILABLE:
            return False
        return self.is_trained

    def _init_model(self):
        """根据 model_type 创建分类器"""
        if self.model_type == "xgb" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

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
        """根据未来收益生成阶段标签"""
        labels = []
        for i in range(len(future_returns) - 1):
            current_ret = future_returns[i]
            next_ret = future_returns[i + 1] if i < len(future_returns) - 1 else 0

            if next_ret >= threshold:
                labels.append(2)
            elif next_ret < -0.01:
                labels.append(4)
            elif current_ret > threshold and next_ret < current_ret * 0.5:
                labels.append(3)
            else:
                labels.append(1)
        return np.array(labels)

    def train_from_backtest(self, backtest_signals: pd.DataFrame) -> dict:
        """从回测信号数据训练模型"""
        if not SKLEARN_AVAILABLE:
            return {"status": "sklearn_not_available", "samples": 0}

        if len(backtest_signals) < self.min_samples:
            return {"status": "insufficient_data", "samples": len(backtest_signals)}

        X_list = []
        y_list = []
        for _, row in backtest_signals.iterrows():
            fwd_ret = row.get("8周", 0)
            if pd.isna(fwd_ret):
                continue

            features = {}
            for col in self.FEATURE_COLS:
                val = row.get(col, 0)
                features[col] = val if not pd.isna(val) else 0
            X_list.append(features)

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

        # XGBoost 要求标签从 0 开始连续
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._init_model()
        self.model.fit(X_scaled, y)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(zip(
                self.FEATURE_COLS,
                [round(v, 3) for v in self.model.feature_importances_]
            ))

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
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
        """对单一样本预测阶段"""
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

            stage_probs = {"accumulation": 0.0, "rising": 0.0, "top": 0.0, "falling": 0.0}
            stage_names = {1: "accumulation", 2: "rising", 3: "top", 4: "falling"}

            for i, cls in enumerate(self.model.classes_):
                orig_label = self._label_encoder.inverse_transform([int(cls)])[0]
                name = stage_names.get(orig_label, "unknown")
                if name in stage_probs:
                    stage_probs[name] = round(float(probs[i]), 3)

            ml_stage = int(self._label_encoder.inverse_transform(
                [int(self.model.predict(features_scaled)[0])]
            )[0])
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
        """融合ML和规则系统的阶段判断"""
        if not ml_result.get("available", False):
            return fuzzy_result

        fuzzy_probs = fuzzy_result.get("stage_probs", {})
        ml_probs = ml_result.get("probs", {})

        fused_probs = {}
        for stage_name in ["accumulation", "rising", "top", "falling"]:
            fp = fuzzy_probs.get(stage_name, 0)
            mp = ml_probs.get(stage_name, 0)
            fused_probs[stage_name] = round(fp * (1 - ml_weight) + mp * ml_weight, 3)

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
        if not SKLEARN_AVAILABLE:
            return {"available": False, "reason": "sklearn 未安装"}
        if self.model_type == "xgb" and not XGBOOST_AVAILABLE:
            return {"available": False, "reason": "xgboost 未安装"}
        if not self.is_trained:
            return {"available": False, "reason": "未训练"}
        return {
            "available": True,
            "trained": True,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance
        }
