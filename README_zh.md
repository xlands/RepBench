# RepBench：将基准编译为大语言模型的能力表征

[English](README.md) | **中文**

**[arXiv 论文](https://arxiv.org/abs/2607.28008)** · 把*任意*评测基准编译成干净的按能力组织的表征数据——并度量能力方向在跨基准、跨模型、跨探针方法下的泛化能力。

大多数表征工程工作从单一数据集里提取"能力方向"，方向因此继承了那个数据集的格式和 token 癖好。本项目补上缺失的数据层：一个从基准文献中挖掘出的能力分类体系、一个**每个能力都有 ≥ 2 个独立基准支撑**的探针语料库，以及一套评测——证明跨基准池化正是让能力向量变干净的关键，对任意模型、任意探针方法都成立。

---

## 1 · 数据管线：开源、可复用的闭环引擎

![RepBench 数据管线](asset/fig1_pipeline.svg)

给定任何一个新基准，管线自动完成：爬取 → 用 10 个模型的隐状态投票 + 人工裁决审计每一条"文本↔能力"映射 → 探针测试 → 把暴露出的缺口反馈给爬取环节——这是一个可重复运转的闭环，而不是一次性的数据集。产出是**方法无关**的：同一份干净数据可以直接喂给 DiffMean、PCA/LAT、CAA、线性探针、SAE、J-Lens、ReFT-r1 等任何方法。

更新后的总览图展示了论文使用的完整闭环：编译 → 审计 → 探针 → 定向补爬。

## 2 · 能力全景图：基准到底在测什么

![Capability landscape](asset/fig2_capability_map.svg)

我们爬取了 **13,427 篇基准论文**，抽取 **14,896 条能力表述**，去重后得到 **9,576 个能力概念**（上图每个点，UMAP 布局，点的大小 ∝ 提及次数），聚成 **13 个能力族、182 个能力簇**（八个最大的族分别着色——多模态理解、推理、代码与调试、安全与鲁棒性、规划与工具使用、事实性与依据、社交与语用、多语言——其余折入灰色）。标签标出最大的簇。182 个簇中有 **94 个**拥有足够的纯文本基准可供探针测试；多模态理解（0/31）和规划与工具使用（0/23）需要图像输入或智能体轨迹，我们把它们如实报告为纯文本探针的覆盖边界。

图中颜色表示分类体系的能力族，点的大小与基准文献中的提及次数成正比。

## 3 · 干净的能力表征：池化前 → 池化后

![Qwen 模型中的池化表征](asset/fig3_representation_models_qwen.svg)

![其他模型中的池化表征](asset/fig3_representation_models_other.svg)

![跨基准池化前后的表征](asset/fig3_representation_pooling.svg)

单个基准的隐状态携带着该基准自己的指纹。由于语料库里每个能力都被 **≥ 2 个基准（中位数 3 个）**覆盖，我们可以把每个能力的表征*跨基准*取平均——基准特有的方差相互抵消，留下一个干净的能力向量。

上面两组图展示六个 checkpoint 的 94 个池化能力向量；下图对比 Qwen3-8B 在池化前后。原始逐文本向量糊成一团，而池化向量在小 k（4–15）处出现内部 silhouette 峰值，且这一点在所有被测模型中都成立。池化面板的颜色是模型发现的簇，原始向量的颜色是人工分类中的能力族。

## 4 · 评测：同一协议下的模型 × 探针方法

![Method × model evaluation](asset/fig4_method_model_eval.svg)

每个（能力，模型，方法）单元格都用**留一基准（leave-one-benchmark-out, LOBO）**评测：方向在完全不含被留出基准任何文本的情况下训练，再在该基准上测试——测的是跨基准泛化，不是记忆。每个模型在四个相对深度（25/50/75/100%）中取自己的最优层。

| | diff-mean | 线性探针（LR） | PCA | J-Lens |
|---|---:|---:|---:|---:|
| 平均 LOBO AUC（1,128 个单元格） | **0.778** | 0.769 | 0.734 | 0.650 |
| 单元格胜率 | 30% | **38%** | 17% | 15% |

两种汇总视角支持不同的读出方式：**diff-mean 的总体均值最高**（且在 12 个模型中的 10 个上最高），而 **LR 赢下最多单元格**。PCA 落后于两个使用标签的激活空间读出；J-Lens 作为检测器较弱，但提供了其他方法没有的、带语义名称的 token 索引接口。R1 蒸馏版 Qwen3-8B 和 DeepSeek-V4-Flash 基座模型是 LR 在模型级均值上最高的两个例外。

**被测模型（12 个）：** Qwen3-0.6B / 1.7B / 4B / 8B / 32B、Qwen3.5-9B、Llama-3.1-8B-Instruct、Gemma2-9B、Gemma4-12B / 31B、DeepSeek-R1-0528-Qwen3-8B（在 R1 轨迹上蒸馏的 Qwen3-8B——作为后训练对照引入，不算额外架构）、DeepSeek-V4-Flash-Base（带 hyper-connection 残差流的 fp8 MoE；按原始基座模型探测——不加对话模板，四条并行残差流取平均，fp8 反量化为 bf16）。

图中每种方法都取其最佳有效观测深度；每个模型的具体数值和层号见 [论文](https://arxiv.org/abs/2607.28008)。

---

## 数据卡

| | |
|---|---|
| 基准数据集 | 353（爬取约 378，经 10 模型一致性投票清洗：标记 64、移除 25） |
| 探针文本 | 46,149 |
| 能力簇 | 94，**100% 由 ≥ 2 个基准支撑**（中位数 3，最多 21） |
| 能力族 | 13 |
| 模型 × 层 | 12 个模型 × 4 个相对深度（最后 token 隐状态） |
| 协议 | 留一基准 AUC，分层负采样 |

## 如实说明的局限

- **覆盖：** 多模态理解、规划与工具使用不在探针语料内（需要图像/智能体输入）——分类体系精确量化了这个缺口有多大。
- **结构结论：** 我们只主张池化后涌现*粗粒度*离散结构（小 k 处的内部 silhouette 峰值），从不主张"恰好 N 个簇"——最优 k 随模型在 4–15 间变化。
- **与人类分类的对齐：** 模型发现的簇不复现人类 13 族分类（ARI ≈ 0.1）。模型按自己的方式组织能力。
- **方法排名取决于统计量：** diff-mean 赢均值（12 个模型中的 10 个），LR 赢单元格计数——且在两个非标准模型（R1 蒸馏版和 V4 基座 MoE）上 LR 连均值也赢。我们两个统计量都报告。

## 仓库结构

```
asset/                          README 使用的矢量图（SVG）
doc/figures/                    早期交互图版本
src/representation_cluster/     表征拟合与评估管线
  scripts/                      语料构建、隐状态抽取、LOBO 读出、
                                SAE、J-Lens 与结果汇总
  results/                      固定的深度扫描与方法结果
  requirements.txt              管线依赖
data/                           公开的 46,149 条探针语料与 manifest
```

## 复现

```bash
python3 -m pip install -r src/representation_cluster/requirements.txt

# 1. 抽取最后 token 隐状态（每张 GPU 一个分片）
CUDA_VISIBLE_DEVICES=0 python3 src/representation_cluster/scripts/extract_hidden.py --model <hf_path> --tag mymodel --shard 0 --nshards 4
#    大型 fp8 权重：加 --device-map auto --dequant-fp8

# 2. diff-mean 探针 + 最优层选择（LOBO）
python3 src/representation_cluster/scripts/probe_clusters.py --tag mymodel

# 3. 最优层上的方法对比（diff-mean / LR / PCA）
python3 src/representation_cluster/scripts/method_compare.py --tag mymodel

# 4. 汇总各方法的最佳观测深度
python3 src/representation_cluster/scripts/summarize_four_method_best_depth.py
```

## 引用

```bibtex
@article{li2026repbench,
  title   = {RepBench: Compiling Benchmarks into Capability Representations
             for Large Language Models},
  author  = {Li, Yanshi and Bai, Xueru and Liu, Shuman and Zhang, Long},
  journal = {arXiv preprint arXiv:2607.28008},
  year    = {2026}
}
```
