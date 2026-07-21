# 受试者级全局归一化：窗口准确率检查与 V2 提升方案

## 1. 已有结果

下表均使用每名受试者全部预处理连续记录的逐通道统计量，训练、验证和测试窗口共享该受试者统计量。它不是三个集合分别拟合，也不是样本级归一化。

| 窗口 | 重叠率（发作/非发作） | 完成折数 | Accuracy | AUROC | 结论 |
|---|---:|---:|---:|---:|---|
| 1 s | 50% / 50% | 3/5 | 96.06% ± 0.30% | 99.23% | CUDA 中断，不能作为正式五折结论 |
| 2 s | 50% / 50% | 5/5 | 96.40% ± 0.44% | 99.30% | 当前最佳完整结果 |
| 4 s | 25% / 0% | 5/5 | 95.51% ± 0.53% | 99.01% | 准确率最低，训练/测试差距明显 |

多数折在最佳 epoch 的训练 Accuracy 已达到 99%–100%，但测试 Accuracy 仍约 95%–96%。因此主要问题不是模型容量不足，而是过拟合、固定阈值未校准、不同窗长共用同一感受野，以及专家分支没有直接参与最终预测。

## 2. V2 改动

V2 保留原始模型和配置，使用独立模型名 `tcn_mse_margat_v2`，输出不会覆盖旧实验。

1. **窗长适配的 TCN 感受野**：1 s、2 s、4 s 分别使用 4、5、6 层，感受野约为 0.71 s、1.46 s、2.96 s。1 s 不再让高层卷积主要读取补零，4 s 则获得更长上下文。
2. **注意力 + 统计量池化**：将注意力池化、全局均值和标准差联合映射，降低只依赖少数高权重时刻造成的过拟合。
3. **四路证据融合**：最终 logit 直接融合主分类器、TCN、MSE、MARGAT 四路输出。旧模型的三个专家头只计算辅助损失，不能直接改变最终决策。
4. **提高有效融合门**：安全残差门初值从 0.02 调到 0.10，仍保持 TCN 为主，但避免 MSE/MARGAT 在训练早期几乎被关闭。
5. **泛化正则**：标签平滑 0.05、权重衰减 `1e-3`、更高 dropout，并以验证集 AUROC 早停。
6. **验证集阈值校准**：每折只用该折验证集选择 Accuracy 最优阈值，再应用于测试集。报告同时保存 `fixed_threshold_accuracy`，可区分结构提升与阈值提升，测试标签不参与选阈值。

## 3. 三组正式五折命令

以下命令直接使用已有 window index、split 目录和预先计算的全局统计缓存，不需要重新生成 processed 数据。运行目录自动包含窗口标识、V2 模型目录和微秒级时间戳。

```bash
python scripts/run_mixed_5fold.py --config configs/normalization/subject_global_win1s_ov50.yaml --model_config configs/model_tcn_mse_margat_v2_win1s.yaml
```

```bash
python scripts/run_mixed_5fold.py --config configs/normalization/subject_global_win2s_ov50.yaml --model_config configs/model_tcn_mse_margat_v2_win2s.yaml
```

```bash
python scripts/run_mixed_5fold.py --config configs/normalization/subject_global.yaml --model_config configs/model_tcn_mse_margat_v2_win4s.yaml
```

## 4. 验收方式与后续优先级

- 正式结论必须来自每组完整 5 折；当前 1 s 的 3 折旧结果只作诊断参考。
- 同时比较 `accuracy` 与 `fixed_threshold_accuracy`：前者包含合法的验证集校准，后者用于和旧模型固定 0.5 阈值公平对照。
- 若 V2 仍未达到目标，优先做 V2 的 `TCN+MSE（无图）` 对照。旧 4 s 消融中图分支没有稳定提升，不能默认增加图层一定更好。
- 99% 以上不能由现有结果保证。若只有通过相邻重叠窗口跨训练/测试集合才能达到 99%，该结果属于切片泄漏而非泛化提升，应另外报告按原始记录或发作事件分组的无泄漏交叉验证。

## 5. 验证状态

- 全部 60 项单元测试通过。
- 1 s、2 s、4 s V2 前向结构、专家权重归一化、窗长层数、阈值选择、训练报告字段均已测试。
- 当前 GPU 高负载且温度较高，本次没有启动耗时的三组五折训练，因此本文没有声称 V2 已达到 99%。
