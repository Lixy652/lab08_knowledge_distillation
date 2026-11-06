# 实验作业八：自制语言模型的知识蒸馏（纯 CPU）

用字符级 Transformer 从零训练一个 0.74 M 参数的"教师"，再把它蒸馏成 0.15 M 参数的
"学生"，在纯 CPU 上完成受控对照实验、T×α 二维消融与 5 项拓展实验。

* 单文件实现，**零外部数据、零预训练权重、无需联网**
* 语料由代码按固定随机种子现场生成（"仿古诗句"合成语料 + 16 首唐诗）
* 教师 / 学生结构完全相同，只有嵌入宽度不同（160 vs 64）

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `distill_lm.py` | 全部实验代码（语料、分词、GPT、训练、蒸馏、消融、拓展） |
| `requirements.txt` | 依赖（仅 CPU 版 PyTorch，可选 matplotlib） |
| `作业八_自制语言模型知识蒸馏_实验报告.docx` | 实验报告（含数据表、曲线、分析、思考题） |

运行产物统一写入 `runs_distill/`（首次运行自动创建）：

| 产物 | 内容 |
| --- | --- |
| `teacher.pt` / `teacher_c96.pt` / `teacher_c256.pt` | 教师权重缓存（已加入 `.gitignore`） |
| `student_*.pt` | 学生权重（已加入 `.gitignore`） |
| `results_*.json` | 每次运行的全部指标 |
| `fig1_train_curves.png` / `fig2_ablation.png` | 报告里的图 1、图 2 |
| `curves.png` / `heatmap.png` | 训练曲线、4×4 消融热力图 |

## 环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib
```

> 务必带 `--index-url .../whl/cpu`，否则可能下载 2 GB 以上的 CUDA 版。
> 实测版本：Python 3.12 / PyTorch 2.14.1+cpu / matplotlib 3.11.2，线程数固定为 8。

Windows 控制台如果出现中文乱码，先 `set PYTHONIOENCODING=utf-8`（或 `chcp 65001`）。

## 复现步骤

```bash
python distill_lm.py --mode quick              # 约 15 s 自检，结果无参考价值
python distill_lm.py --mode teacher --retrain  # 训练教师，约 40 s
python distill_lm.py --mode all                # 从零学生 vs 蒸馏学生，约 30 s
python distill_lm.py --mode ablate             # T x alpha 4x4 消融，约 100 s
python distill_lm.py --mode figures            # 合成报告用的图 1 / 图 2
```

拓展实验（对应报告第八节 A~E）：

```bash
python distill_lm.py --mode feat --feat 0.1    # A 特征蒸馏 FitNet
python distill_lm.py --mode heatmap            # B 消融热力图
python distill_lm.py --mode selfdistill        # C 自蒸馏（教师 = 训练更久的同构学生）
python distill_lm.py --mode quant              # D 动态量化 int8
python distill_lm.py --mode teacherscan        # E 教师规模扫描 C in {96,160,256}
```

辅助/自检模式：

```bash
python distill_lm.py --mode sanity             # KL 未 reshape 的放大倍数、T^2 对梯度的作用
python distill_lm.py --mode randteacher        # 对照：教师换成未训练的随机网络
python distill_lm.py --mode all --T 100 --run-name _T100   # 温度宽扫描（结果写 results_all_T100.json）
```

常用参数：`--teacher-iters 200 --student-iters 120 --abl-iters 80 --batch-size 24
--lr 3e-3 --T 4 --alpha 0.5 --seed 0`。全部实验串行跑完约 12 分钟。

## 蒸馏损失

```
L = (1 - alpha) * CE(y, p_s) + alpha * T^2 * KL(p_t^T || p_s^T)
```

实现上有三个必须注意的点（代码中均有注释，`--mode sanity` 可现场验证）：

1. `KL` 前先把 logits 由 `(B, T, V)` reshape 成 `(B*T, V)`，否则
   `kl_div(reduction="batchmean")` 只除以 batch，序列维被重复累加，损失被放大 96 倍
   （实测 5.5281 → 530.6932，正好等于 `block_size`）；
2. `KL` 项乘 `T^2`，抵消 `softmax(z/T)` 在梯度上带来的 `1/T` 因子：不乘时 T 从 1 到 8
   会让 KD 梯度缩到约 1/320，等于把蒸馏关掉；
3. `KL` 的方向是 `KL(教师 || 学生)`：教师分布作目标 `target`，学生分布取 `log`。

对照实验的公平性：`scratch` 与 `kd` 两组学生用同一随机种子初始化（`seed + 7`）、
同一个数据采样器（`seed + 1`）喂同样的批次，唯一区别是 `kd` 多了一项 KL。

## 结果概览（本机实测）

| 模型 | 参数量 | 留出集 loss | PPL | 与教师一致率 |
| --- | --- | --- | --- | --- |
| 教师 Teacher | 0.737 M | 0.8322 | 2.30 | — |
| 学生·从零训练 | 0.147 M | 1.2722 | 3.57 | 0.613 |
| 学生·蒸馏训练 | 0.147 M | 1.2107 | 3.36 | 0.656 |

压缩 5.00 倍，留出集损失相对下降 4.84%，与教师一致率从 0.613 提升到 0.656。

其他结论（详见报告）：

* **T×α 网格**最优 T=8/α=0.3（1.4008），最差 T=8/α=0.9（1.6722）；T=1 时四组几乎相同（极差 0.0024）。
* 固定 α=0.5、120 步时，温度一直加到 T=100 反而最好（1.1859）：乘了 T² 之后，大 T 极限下
  KD 退化为"中心化 logits 的 L2 匹配"，所以"温度过高失效"只在漏乘 T² 时成立。
* **随机教师对照**（教师不训练）：1.4073，比从零训练还差 → 收益来自教师的暗知识而非单纯正则。
* **教师规模扫描**：C=96 → 2.4%、C=160 → 4.8%、C=256 → 1.2%，并非单调，存在 capacity gap。
* **自蒸馏**：教师换成训练 400 步的同构学生（0.147 M，0.8475），收益 2.1%（真教师是 4.8%）。
* **动态量化**：674,384 B → 324,222 B（2.08×），留出集 loss 1.2107 → 1.2115；
  但 batch=1 下单次前向 1.74 ms → 3.49 ms，省的是存储不是延迟。
