# CRUNCH

**Corn End-Use Recognition Using Normalized Indices, Change Points, and Harmonics**

作者：**TIANYANG LIU 和 BORAN CUI**。

## 使用

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python evaluate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py --retrain
```

`train.py` 从训练集原始有效指数观测重训；`predict.py` 对新数据推理；`evaluate.py` 输出逐例预测和指标。无需 GPU 或网络采集，依赖安装后离线运行。

测试集 Accuracy **95.45%（42/44）**，Macro-F1 **0.939394**，平衡准确率 **0.916667**。测试集的44个田块年份来自34个物理田块。MD未参与拟合，但此前研究已反复查看其结果，不能称作全新盲测。

## 文档

- [英文 README 与完整命令](README.md)
- [架构与特征规则](docs/ARCHITECTURE.md)
- [数据集与来源](docs/DATASET.md)
- [模型卡](docs/MODEL_CARD.md)
- [与 MiniRocket / MultiRocket / ROCKET / TempCNN 的比较及论文引用](docs/BENCHMARKS.md)

代码、团队文档及模型权重采用 [MIT License](LICENSE)。数据保留上游来源条款；WA标签与MD比赛子集的明确再分发许可在现有材料中尚未确认，详见 [DATA_LICENSE.md](DATA_LICENSE.md)。此包包含请求的训练与测试快照，但不把第三方数据重新声明为团队 MIT 数据。
