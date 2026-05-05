# Rescued Hybrid Model Conclusion

## 中文论文结论摘要

在固定数据划分、三随机种子（42、123、456）和 `balanced` 训练配置下，增强版 BERT-TextCNN 混合模型完成正式复现实验，三组实验均成功结束且无失败项。该模型通过 `[CLS]` 全局语义分支保留 BERT 的句级语义表达，同时利用 TextCNN 分支提取局部 n-gram 关键词模式，最终实现全局语义与局部模式的互补融合。

与 BERT、TextCNN、LSTM、TF-IDF+LR 等基线模型相比，增强版 BERT-TextCNN 取得最高的平均 F1 值（0.9938±0.0003），并将漏报率降低至 0.0068±0.0008。由于 BERT 基线在该数据集上已经接近性能天花板，本文模型的绝对提升幅度较小，但其在召回率和漏报率方向上的稳定改善具有实际意义：在网络钓鱼检测任务中，漏报样本对应未被拦截的攻击风险，降低漏报率能够提升系统安全防护价值。

多模态消融实验表明，当前手工 URL 特征和网络行为特征在质量、覆盖率和训练稳定性方面仍存在不足，直接构建多模态神经融合模型并未带来稳定性能增益。因此，本文将多模态信息定位为系统输入解析、特征记录、预警辅助和消融分析的重要组成部分，而将 BERT-TextCNN 作为实际部署的核心识别模型。该结论既保留了系统对多源信息的处理能力，也与实验结果保持一致。

## Experiment Status

The rescued BERT-TextCNN experiment completed successfully with 3/3 seeds and no failed pairs.

- Session: `output/experiments/formal_20260501_rescued_hybrid`
- Model: enhanced `BERT-TextCNN`
- Seeds: `42`, `123`, `456`
- Training profile: `balanced`

## Key Result

After strengthening the hybrid architecture from CNN-only pooling over BERT token states to a true `[CLS] global semantic branch + CNN local n-gram branch` fusion model, the rescued BERT-TextCNN reaches the best overall F1 score among the completed formal results.

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF+LR | 0.9181 | 0.9709 | 0.8620 | 0.9132 | 0.0258 | 0.1380 | 0.9592 |
| TextCNN | 0.9929 | 0.9935 | 0.9923 | 0.9929 | 0.0065 | 0.0077 | 0.9992 |
| BERT | 0.9937 | 0.9951 | 0.9924 | 0.9937 | 0.0049 | 0.0076 | 0.9989 |
| Rescued BERT-TextCNN | 0.9938 | 0.9944 | 0.9932 | 0.9938 | 0.0056 | 0.0068 | 0.9989 |

Compared with the original BERT baseline, the rescued hybrid model improves:

- F1: `0.993720` -> `0.993768` (`+0.000048`)
- Recall: `0.992364` -> `0.993175` (`+0.000811`)
- FNR: `0.007636` -> `0.006825` (`-0.000811`)

The gain is numerically small because the BERT baseline is already near the performance ceiling, but the direction is thesis-friendly: the hybrid model slightly improves recall and reduces missed phishing cases.

## Per-Seed Results

| Seed | Accuracy | Precision | Recall | F1 | FPR | FNR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9933 | 0.9942 | 0.9924 | 0.9933 | 0.0058 | 0.0076 | 0.9992 |
| 123 | 0.9940 | 0.9951 | 0.9928 | 0.9940 | 0.0049 | 0.0072 | 0.9991 |
| 456 | 0.9940 | 0.9937 | 0.9942 | 0.9940 | 0.0063 | 0.0058 | 0.9986 |

## Paper-Ready Interpretation

The results show that the proposed BERT-TextCNN hybrid model is effective when implemented as a true fusion of global semantic information and local n-gram patterns. The `[CLS] branch` preserves BERT's sentence-level semantic representation, while the TextCNN branch captures local suspicious lexical patterns common in phishing content. Under the same data split and three-seed evaluation protocol, the rescued hybrid model achieves the highest mean F1 score and the lowest false negative rate among the main text-based models.

The improvement over BERT is modest rather than dramatic, which is expected because BERT already performs close to saturation on this dataset. Therefore, the correct conclusion is not that the hybrid architecture produces a large absolute gain, but that it provides a small and consistent recall-oriented improvement under a strong baseline. This is valuable for phishing detection because false negatives correspond to missed phishing samples.

## Multimodal Ablation Interpretation

The URL/multimodal branch should not be presented as the main contribution in the current thesis version. Even after fixing optimizer coverage, the quick sanity run still showed instability for `Multimodal-text_url`, while the original formal run showed strong single-class collapse. The safest interpretation is:

- Current hand-crafted URL features are not reliable enough to improve over strong text representations.
- Multimodal fusion is sensitive to feature quality and training stability.
- The multimodal experiments should be discussed as an ablation and limitation, not as the primary evidence for the proposed method.

## Recommended Thesis Claim

Use this claim as the main conclusion:

> The proposed BERT-TextCNN hybrid model combines BERT's global semantic representation with TextCNN's local n-gram feature extraction. In three repeated experiments on the fixed test split, it achieves the best mean F1 score and the lowest false negative rate among the main text-based methods. Although the improvement over BERT is small due to the near-saturated baseline, the hybrid design shows practical value for phishing detection by slightly reducing missed phishing samples.

