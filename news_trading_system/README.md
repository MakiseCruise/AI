# News Trading System — Home Task 3

## 概述

一个基于新闻情感分析的简单交易系统。该系统：

1. 从 Sina 财经（通过 akshare）获取 S&P 500 指数的年度价格历史
2. 获取金融新闻（优先使用 Google News RSS，备选内置新闻数据集）
3. 使用 NLTK VADER 进行情感分析
4. 根据每日聚合情感分数生成 Buy/Sell/Hold 信号
5. 计算年化收益率（起始资本的百分比）
6. 与 Buy & Hold 策略对比，生成可视化对比图

## 安装依赖

```bash
pip install akshare feedparser nltk pandas numpy matplotlib
```

## 运行

```bash
cd news_trading_system
python news_trading_system.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `comparison_chart.png` | 三条面板的对比图（收益曲线 + 持仓 + 情感） |
| `backtest_results.csv` | 每日策略收益 vs Buy&Hold 收益详表 |
| `news_with_sentiment.csv` | 每条新闻标题及其 VADER 情感得分 |

## 数据源

| 数据类型 | 来源 | 备注 |
|----------|------|------|
| 价格数据 | Sina 财经（通过 akshare） | 免费，无需 API Key，国内可访问 |
| 新闻数据 | Google News RSS | 自动尝试在线获取，失败时使用内置 fallback 数据集 |
| 情感分析 | NLTK VADER | 基于词典的情感分析器 |

## 策略逻辑

- **Buy**：日聚合 VADER compound > +0.05 → 次日全仓做多
- **Sell**：日聚合 VADER compound < -0.05 → 次日全仓做空
- **Hold**：介于两者之间 → 维持当前仓位

**注意**：系统严格遵守"无前瞻偏差"原则 —— 每个交易日的信号仅基于该日之前发布的新闻。

## 测试结果（2025-05-19 → 2026-07-22）

| 指标 | News Strategy | Buy & Hold |
|------|:------------:|:----------:|
| 总收益 | -14.35% | +25.75% |
| 年化夏普比率 | -1.01 | 1.65 |
| 最大回撤 | -25.47% | -9.10% |
| 最终价值 | $85,654 | $125,746 |

## 分析

News Sentiment 策略在此测试区间**跑输** Buy & Hold 约 40 个百分点。

**原因**：
- VADER 是面向社交媒体的通用情感工具，缺乏金融领域的专业理解
- 新闻标题通常较短（5-12 词），情感信号较弱
- 策略频繁翻转多空仓位，在震荡行情中产生"鞭打效应"
- S&P 500 在此期间呈强劲上涨趋势，Buy & Hold 天然受益
- 短暂回调期间做空，市场快速反弹造成损失

**改进方向**：
- 用 FinBERT 替代 VADER（金融领域微调的 BERT 模型）
- 使用更高质量的新闻源（Bloomberg/Reuters API）
- 添加趋势过滤（仅在宏观趋势看跌时做空）
- 优化阈值参数
- 加入交易成本建模
