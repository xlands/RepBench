# 潜在能力表征基准（Latent Capability Representation Benchmark）

[English](README.md) | **中文**

**把*任意*评测基准编译成干净的按能力组织的表征数据——并度量能力方向在跨基准、跨模型、跨探针方法下的泛化能力。**

大多数表征工程工作从单一数据集里提取"能力方向"，方向因此继承了那个数据集的格式和 token 癖好。本项目补上缺失的数据层：一个从基准文献中挖掘出的能力分类体系、一个**每个能力都有 ≥ 2 个独立基准支撑**的探针语料库，以及一套评测——证明跨基准池化正是让能力向量变干净的关键，对任意模型、任意探针方法都成立。

---

## 1 · 数据管线：开源、可复用的闭环引擎

![Data pipeline](asset/fig1_pipeline.png)

给定任何一个新基准，管线自动完成：爬取 → 用 10 个模型的隐状态投票 + 人工裁决审计每一条"文本↔能力"映射 → 探针测试 → 把暴露出的缺口反馈给爬取环节——这是一个可重复运转的闭环，而不是一次性的数据集。产出是**方法无关**的：同一份干净数据可以直接喂给 DiffMean、PCA/LAT、CAA、线性探针、SAE、J-Lens、ReFT-r1 等任何方法。

*交互版：[`doc/figures/pipeline.html`](doc/figures/pipeline.html)*

## 2 · 能力全景图：基准到底在测什么

![Capability landscape](asset/fig2_capability_map.png)

我们爬取了 **13,427 篇基准论文**，抽取 **14,896 条能力表述**，去重后得到 **9,576 个能力概念**（上图每个点，UMAP 布局，点的大小 ∝ 提及次数），聚成 **13 个能力族、182 个能力簇**（八个最大的族分别着色——多模态理解、推理、代码与调试、安全与鲁棒性、规划与工具使用、事实性与依据、社交与语用、多语言——其余折入灰色）。标签标出最大的簇。182 个簇中有 **94 个**拥有足够的纯文本基准可供探针测试；多模态理解（0/31）和规划与工具使用（0/23）需要图像输入或智能体轨迹，我们把它们如实报告为纯文本探针的覆盖边界。

*交互版（缩放 / 搜索 / 按族筛选）：[`doc/figures/capability_map.html`](doc/figures/capability_map.html)*

## 3 · 干净的能力表征：池化前 → 池化后

![Representation clustering](asset/fig3_representation_clusters.png)

单个基准的隐状态携带着该基准自己的指纹。由于语料库里每个能力都被 **≥ 2 个基准（中位数 3 个）**覆盖，我们可以把每个能力的表征*跨基准*取平均——基准特有的方差相互抵消，留下一个干净的能力向量。

**(A)** 六个模型各自把 94 个池化后的能力向量聚成少数几个分离清晰的组。**(B)** 同一模型池化前后对比：46,149 条原始逐文本向量糊成一团，而 94 个池化向量干净地分开。定量上：原始向量的 silhouette-vs-k 曲线单调爬升到扫描上限（不存在自然簇数），而池化后在小 k 处（4–15）出现内部峰值——**所有被测模型无一例外**。我们只主张"粗粒度结构的涌现"，不主张确切的簇数；模型自己发现的簇并不复现人类的能力族分类（ARI ≈ 0.1）——这在预期之内，而且本身就是有趣的发现。

*交互版：[`doc/figures/representation_clusters.html`](doc/figures/representation_clusters.html)*

## 4 · 评测：同一协议下的模型 × 探针方法

![Method × model evaluation](asset/fig4_method_model_eval.png)

每个（能力，模型，方法）单元格都用**留一基准（leave-one-benchmark-out, LOBO）**评测：方向在完全不含被留出基准任何文本的情况下训练，再在该基准上测试——测的是跨基准泛化，不是记忆。每个模型在四个相对深度（25/50/75/100%）中取自己的最优层。

| | diff-mean | 线性探针（LR） | PCA |
|---|---|---|---|
| 平均 LOBO AUC（全部模型） | **0.778** | 0.764 | 0.724 |
| 单元格胜率 | 36% | **47%** | 17% |

诚实的解读：**diff-mean 在 12 个模型中的 10 个上均值最高**（下限最高、免训练——最好的单一默认方法），而 **LR 赢下最多单元格**（它在接近天花板的简单簇上小胜，但在困难簇上崩得更狠）。两个例外恰好是两个非标准模型：R1 蒸馏版 Qwen3-8B（LR 0.754 vs 0.732）和 DeepSeek-V4-Flash 基座模型（LR 0.733 vs 0.720）。蒸馏这组是受控对照——架构与 Qwen3-8B 完全相同，蒸馏却让 diff-mean 下降 0.053（从 0.785）——暗示推理后训练会重塑能力的线性编码方式。方法轴的差异（~0.05）仍然大于模型轴（~0.02）——探针方法本身就是一个有意义的评测维度。

**被测模型（12 个）：** Qwen3-0.6B / 1.7B / 4B / 8B / 32B、Qwen3.5-9B、Llama-3.1-8B-Instruct、Gemma2-9B、Gemma4-12B / 31B、DeepSeek-R1-0528-Qwen3-8B（在 R1 轨迹上蒸馏的 Qwen3-8B——作为后训练对照引入，不算额外架构）、DeepSeek-V4-Flash-Base（带 hyper-connection 残差流的 fp8 MoE；按原始基座模型探测——不加对话模板，四条并行残差流取平均，fp8 反量化为 bf16）。

*交互版：[`doc/figures/method_model_eval.html`](doc/figures/method_model_eval.html)*

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
asset/                          README 使用的渲染图（PNG）
doc/figures/                    四张图的交互式 HTML 版本
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
@article{latent_capability_benchmark_2026,
  title   = {A Latent Capability Representation Benchmark: Clean Per-Capability
             Data for Representation Engineering},
  author  = {...},
  journal = {Under review},
  year    = {2026}
}
```
