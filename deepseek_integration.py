"""
DeepSeek API 集成模块

功能：
1. 信号交叉验证 - DeepSeek对技术指标给出第二意见
2. 市场情绪分析 - 基于新闻/公告的情绪判断（需输入文本）
3. 投资解释报告 - 生成自然语言的决策解释
4. 持仓组合检查 - 分析组合风险

使用方式：
  client = DeepSeekClient(api_key="your-key")
  result = client.validate_signal(metrics, stage)

如果未配置API key，所有函数返回None（优雅降级），不影响主系统运行。
"""
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime


class DeepSeekClient:
    """DeepSeek API 客户端（OpenAI 兼容格式）"""

    def __init__(self, api_key: str = "", model: str = "deepseek-chat", timeout: int = 15):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._available = bool(api_key and len(api_key) > 10)

    @property
    def available(self) -> bool:
        return self._available

    def _call_api(self, messages: list, temperature: float = 0.3, max_tokens: int = 500) -> Optional[str]:
        """调用 DeepSeek API（OpenAI 兼容格式）"""
        if not self._available:
            return None
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"DeepSeek API error: {resp.status_code} {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return None

    def validate_signal(self, stage: int, confidence: float, metrics: dict,
                        market_regime: str = "") -> dict:
        """
        基于技术指标交叉验证当前信号，返回第二意见

        Args:
            stage: 当前阶段 (1-4)
            confidence: 置信度 (0-1)
            metrics: 关键指标字典
            market_regime: 市场状态
        """
        if not self._available:
            return {"deepseek_opinion": None, "note": "DeepSeek未配置，使用规则系统判断"}

        stage_names = {1: "筑底期", 2: "上升期", 3: "顶部震荡期", 4: "下跌期"}
        stage_name = stage_names.get(stage, f"阶段{stage}")

        prompt = f"""你是一个专业的A股技术分析顾问，请基于以下技术指标对当前市场阶段进行判断验证：

## 当前系统判断
- 阶段: {stage_name} (置信度: {confidence:.0%})
- 市场状态: {market_regime or '未检测'}

## 技术指标
- 价格vs30周均线偏离: {metrics.get('diff30_pct', 'N/A')}%
- 价格vs10周均线偏离: {metrics.get('diff10_pct', 'N/A')}%
- 30周均线斜率: {metrics.get('ma30_slope', 'N/A')}
- 30周均线拟合度R²: {metrics.get('ma30_r2', 'N/A')}
- RSI(14): {metrics.get('rsi', 'N/A')}
- 4周涨幅: {metrics.get('ret_4w', 'N/A')}%
- 8周涨幅: {metrics.get('ret_8w', 'N/A')}%
- 均线排列: {'多头' if metrics.get('ma_arrangement') == 1 else '空头' if metrics.get('ma_arrangement') == -1 else '缠绕'}
- 成交量比率: {metrics.get('vol_ratio', 'N/A')}

## 请输出（JSON格式）
{{
  "agree": true/false,
  "alternative_stage": "筑底/上升/顶部/下跌",
  "alternative_confidence": 0-100,
  "key_signals": ["看多的信号1", "看多的信号2"],
  "key_risks": ["看空的信号1", "看空的信号2"],
  "advice": "一句话操作建议"
}}
只输出JSON，不要额外说明。"""
        response = self._call_api([
            {"role": "system", "content": "你是一个A股技术分析师，基于指标做判断，只输出JSON。"},
            {"role": "user", "content": prompt}
        ], temperature=0.2, max_tokens=600)

        if response:
            try:
                parsed = json.loads(response.strip().strip('```json').strip('```').strip())
                return {"deepseek_opinion": parsed, "note": "DeepSeek已验证"}
            except (json.JSONDecodeError, KeyError):
                return {"deepseek_opinion": {"raw": response}, "note": "DeepSeek返回未解析"}
        return {"deepseek_opinion": None, "note": "DeepSeek请求失败"}

    def market_analysis(self, market_regime: dict, top_funds: list = None) -> Optional[str]:
        """
        生成市场分析简报

        Args:
            market_regime: 市场状态检测结果
            top_funds: 评分靠前的基金列表
        """
        if not self._available:
            return None

        regime = market_regime.get("regime", "unknown")
        regime_desc = market_regime.get("description", "")
        score = market_regime.get("score", 0)

        fund_text = ""
        if top_funds:
            fund_text = "\n".join([f"- {f.get('code', '')} {f.get('name', '')} 评分:{f.get('score', 0)} 建议:{f.get('advice', '')}" for f in top_funds[:5]])

        prompt = f"""你是一个A股市场分析师，请基于以下数据进行分析：

## 市场状态
- 状态: {regime}
- 评分: {score}/100
- 描述: {regime_desc}

## 推荐基金
{fund_text or '暂无推荐'}

请给出：
1. 当前市场环境的核心判断（一句话）
2. 操作策略建议（防守/进攻/均衡）
3. 风险提示"""
        return self._call_api([
            {"role": "system", "content": "你是一个专业的市场分析师，给出简洁实用的分析。"},
            {"role": "user", "content": prompt}
        ], temperature=0.5, max_tokens=800)

    def generate_explanation(self, fund_name: str, advice: str, score: float,
                             stage_info: dict, risk_info: dict) -> Optional[str]:
        """
        生成自然语言的投资建议解释
        """
        if not self._available:
            return None

        stage = stage_info.get("stage", 0)
        stage_names = {1: "筑底期", 2: "上升期", 3: "顶部期", 4: "下跌期"}

        prompt = f"""请用一段话解释以下投资建议：

基金: {fund_name}
建议操作: {advice}
综合评分: {score}/100
当前阶段: 第{stage}阶段({stage_names.get(stage, '未知')})
阶段判断: {stage_info.get('reason', '')}

风险数据:
- 最大回撤: {risk_info.get('max_drawdown', 0)}%
- 夏普比率: {risk_info.get('sharpe_ratio', 0)}

请用通俗易懂的中文解释为什么给出这个建议，控制在100字以内。"""
        return self._call_api([
            {"role": "system", "content": "你是投资顾问，用通俗语言解释建议。"},
            {"role": "user", "content": prompt}
        ], temperature=0.5, max_tokens=300)

    def portfolio_check(self, holdings: list) -> Optional[str]:
        """
        检查持仓组合风险

        Args:
            holdings: [{"code": "...", "name": "...", "position_pct": 30, "score": 75}, ...]
        """
        if not self._available:
            return None

        text = json.dumps(holdings, ensure_ascii=False, indent=2)
        prompt = f"""检查以下基金持仓组合的风险：

{text}

请分析：
1. 持仓集中度风险
2. 整体评分评估
3. 调整建议（100字内）"""
        return self._call_api([
            {"role": "system", "content": "你是风控专家，分析持仓风险。"},
            {"role": "user", "content": prompt}
        ], temperature=0.3, max_tokens=400)

    def batch_ranking_advice(self, fund_rankings: list) -> Optional[str]:
        """
        对批量分析结果给出综合建议

        Args:
            fund_rankings: 按评分排序的基金列表
        """
        if not self._available:
            return None

        lines = []
        for f in fund_rankings[:8]:
            lines.append(f"- {f.get('code')} {f.get('name')}: 评分{f.get('score')} 建议{f.get('advice')} 仓位{f.get('position', 0)}%")
        prompt = f"""以下基金按综合评分排序：

{'请根据以上排序给出：1. 哪些基金值得优先考虑 2. 整体配置建议 3. 风险提示（150字内）'}

{chr(10).join(lines)}"""
        return self._call_api([
            {"role": "system", "content": "给出基金排序分析和配置建议。"},
            {"role": "user", "content": prompt}
        ], temperature=0.4, max_tokens=500)
