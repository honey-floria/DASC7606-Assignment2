# DASC7606 Assignment 2 待办清单

截止：**2026 年 11 月 22 日 23:59，香港时间（北京时间）**。

依据仓库 README.md、RUBRIC.md 和 EXTENSION.md（2026-11-17 更新）。核心部分 80 分，扩展 20 分，总计 100 分；不做扩展最高 80 分。最终以 Moodle 和正式作业说明为准。

## 1. 准备环境与数据

- [x] 阅读 README.md、RUBRIC.md、EXTENSION.md，确认 Moodle 提交要求。
- [x] 准备 Python 3.12 或 3.13 独立环境，安装适合硬件的 PyTorch 和 requirements.txt。
- [x] 准备正式训练使用的 CUDA GPU；CPU 可用于正确性测试和 smoke run，现有脚本不支持 MPS。
- [x] 下载指定 v1.0.0 数据包及校验文件，验证 SHA-256 后解压到仓库根目录。
- [x] 确认 data/ 下有 train.bin、val.bin、tokenizer.json、manifest.json 和许可证文件。
- [x] 执行 `python scripts/check_setup.py`，确认环境和数据检查通过。
- [x] 记录实际硬件、Python/PyTorch 版本和精度设置，供报告使用。

约束：不更换数据或 tokenizer，不引入其他训练文本，不使用预训练权重，不在验证集上训练。Baseline 使用规定的 BF16；若环境无法执行，先联系 TA，不自行修改 baseline 精度。

## 2. 补全模型代码（20 分）

文件：`src/model.py`。保留公共类、方法签名及 checkpoint key 名称。

- [x] 实现 `CausalSelfAttention.forward`：Q/K/V 投影、多头拆分、缩放点积、因果 mask、softmax、attention dropout、合并 heads、输出投影和 dropout。
- [x] 显式计算注意力，不使用 SDPA 或 MultiheadAttention 替代。
- [x] 实现 `MLP.forward`：扩维 → GELU → 投影 → dropout。
- [x] 实现 `Block.forward`：Pre-LayerNorm 注意力残差，再用更新后的残差流完成 Pre-LayerNorm MLP 残差。
- [x] 实现 `GPT.generate`：上下文截断、temperature、top-k、使用指定 generator 采样、逐行 EOS 和全部结束时停止。
- [x] 处理 generate 参数校验、无梯度生成、临时 eval 模式，以及异常时也恢复原模式。
- [x] 执行 `python -m pytest -q`，修复失败项。
- [x] 执行 `bash scripts/run_smoke.sh cpu`。
- [x] 在正式 GPU 上执行 `bash scripts/run_smoke.sh cuda`，验证训练环境。

注意：公开测试不覆盖全部评分测试；20 步 smoke run 不算正式 baseline。

## 3. 完成 baseline（20 分）

- [x] 保持 `configs/baseline.yaml` 不变。
- [x] 执行 `bash scripts/train_baseline.sh cuda`，完成 2,500 次 optimizer updates。
- [x] 确认预算为 10,240,000 个 target tokens：2,500 × 16 × 256，gradient accumulation 为 1。
- [x] 保留完整配置、日志、指标和 best.pt；last.pt 仅用于恢复训练，不提交。
- [x] 执行 `python scripts/plot_run.py outputs/baseline`。
- [x] 执行 `python -m src.sample --checkpoint outputs/baseline/best.pt --out outputs/baseline/samples.json`。
- [x] 执行 `python -m src.evaluate --checkpoint outputs/baseline/best.pt --tokens data/val.bin --max-tokens 0 --out outputs/baseline/evaluation.json`。
- [x] 检查曲线与续写，记录学习行为、重复和叙事一致性问题。

Baseline：6 层、6 heads、width 384、context 256、vocab 8192。每个新 run 使用独立输出目录；仅在恢复同配置的中断训练时使用 `--resume`。

## 4. 完成两个受控实验（20 分）

两个实验均相对原 baseline 单独改变一个因素，不在第一个实验上叠加第二个改动。

### 模型侧实验

- [x] 写出研究问题和预期，例如仅改变层数会怎样影响结果。
- [x] 复制 baseline YAML，选定并只修改一个模型因素。
- [x] 保持词表和上下文长度不变，使用支持的 GPT 架构，参数量不超过 20M。
- [x] 使用独立输出目录，完成与 baseline 相同的 processed-token 预算。
- [x] 保存配置、日志、指标；运行绘图、固定 prompt 采样和完整验证集评估。

### 优化侧实验

- [x] 写出研究问题和预期，例如仅改变学习率会怎样影响收敛。
- [x] 另行复制原 baseline YAML，只修改一个优化因素。
- [x] 使用独立输出目录，完成与 baseline 相同的 processed-token 预算。
- [x] 保存配置、日志、指标；运行绘图、固定 prompt 采样和完整验证集评估。

### 比较与选择

- [x] 汇总三个 run 的改动、参数量、token 预算、验证 loss/PPL 和环境信息。
- [x] 使用一致的评估协议与固定 prompt 比较结果。
- [x] 区分训练器抽样混合精度验证 loss 与独立全量 float32 评估结果。
- [x] 分析结果是否支持假设，说明失败案例和可能原因。
- [x] 仅依据公开验证结果，从三个核心 run 中选出最终提交的 best.pt。

没有固定 loss/PPL 得分门槛；公平实验和可信分析比单纯追求更低 loss 更重要。

## 5. 撰写核心报告（20 分）

- [ ] 生成 `report.pdf`：正文不超过 5 页，附录不限页数，无最低页数。
- [ ] 说明模型与四个方法的实现。
- [ ] 说明固定数据、训练设置、实验假设和单因素控制方式。
- [ ] 放入三个 run 的对照表和清晰可读的训练/验证曲线。
- [ ] 放入相同 prompt 下的代表性续写，讨论连贯性、重复、人物和情节一致性。
- [ ] 解释定量结果、负面结果、局限与可合理得出的结论。
- [ ] 写明硬件、软件、精度、种子、token 预算和复现命令。
- [ ] 引用借鉴的方法或实现，明确自己的修改。
- [ ] 检查 PDF 排版、图表和正文页数。

## 6. 可选扩展探索（20 分）

先完成三个核心 run。深入研究一个问题即可；必做实验不能无额外贡献地重复计为扩展。

- [ ] 确定一个有明确动机的方向，例如优化器、位置编码、归一化或注意力改动。
- [ ] 阅读并引用相关来源，说明方法原理、简化之处和自己的贡献。
- [ ] 实现扩展并提供完整可复现的训练、加载、评估命令。
- [ ] 使用相同训练数据、tokenizer、词表和上下文长度，从随机初始化训练，参数不超过 20M。
- [ ] 提供匹配 token 预算和评估协议的对照；必要时做消融。
- [ ] 保存配置、日志、指标、曲线、固定 prompt 续写、运行时间和预算。
- [ ] 分析效果、代价、失败模式和结论边界；负面结果也可得分。
- [ ] 单独撰写 `extension_report.pdf`。
- [ ] 整理 extension/，根目录包含 extension_report.pdf、README.md、code/、configs/、results/。
- [ ] 如需附权重，最多包含一个 extension_best.pt，并说明加载方法；权重可不提交。

扩展按动机与理解、实现、实验证据、分析与洞见各 5 分评分。新架构或组合改动放在扩展包，不替代核心 best.pt。

## 7. 打包、检查和提交

- [ ] 使用 `scripts/package_submission.py`，指定学号、三个 run 目录、选中的 run 和 report.pdf，生成 `学号.zip`。
- [ ] 确认核心包包含代码、配置、测试、必需运行证据、report.pdf 和一个 best.pt；辅助模块放在 src/ 下。
- [ ] 执行 `bash scripts/check_submission.sh 学号.zip` 并修复问题。
- [ ] 不包含数据、虚拟环境、缓存、last.pt、其他 checkpoint 或凭据。
- [ ] 确认核心 ZIP 不超过 100 MB；解压内容不超过 200 MiB，checkpoint 不超过 150 MiB。
- [ ] 如做扩展，执行 `python scripts/package_extension.py --directory extension --student-id 学号 --out 学号_extension.zip`。
- [ ] 执行 `python scripts/package_extension.py --check 学号_extension.zip`。
- [ ] 确认扩展 ZIP 不超过 100 MB，解压内容不超过 200 MiB。
- [ ] 上传核心 ZIP；如有扩展，同时上传独立扩展 ZIP，不替换核心包。
- [ ] 点击最终提交，确认 Moodle 状态是 **Submitted for grading**，而不是草稿。
- [ ] 本地保留最终提交包和完整实验记录。

格式检查通过不代表获得分数，也不替代实现正确性和报告质量检查。
