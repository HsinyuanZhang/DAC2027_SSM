# 2T2R RRAM 权重量化：训练后量化（PTQ）敏感度（2026-10-03）

## 更正（同日，以本节为准）

下文“结果”一节里的 target（跨 session）R² **会掩盖量化损失，不能用来比较量化精度**。判断依据来自 `sanity_checks.py` 和 `srcval_sensitivity.py`：

- **量化确实生效，相关层也确实被用到。** 三值的权重相对误差为 0.40；预测相对浮点预测的 RMS 变化，M1 为 0.30，M2 为 0.48。跳过所有 block，或把 SSM 支路置零，source 验证 R² 会降到 0 附近甚至更低。
- **在没有分布偏移的 source 验证集上，全部三值的损失为 M1 −0.036、M2 −0.048。** 换成每行范数相同的随机噪声，损失与三值相当（M1 −0.024，M2 −0.051）。但同样的随机噪声让 M2 的 target R² **升高了 0.047**。
- **原因**：SSE_new = SSE + ‖Δ‖² − 2⟨e, Δ⟩。跨 session 的残差 e 很大，交叉项 ⟨e, Δ⟩ 的方差远大于 ‖Δ‖²，所以 target R² 的变化被一个符号随机的项主导，可升可降。

source 验证 R² 的结果（INT8 激活，激活尺度只在 source 训练段上标定）：

| 配置 | 2T2R 对数（Mi） | M1 Δ（浮点 0.773） | M2 Δ（浮点 0.502） |
|---|---:|---:|---:|
| 全部 8 bit | 2.24 | −0.001 | −0.002 |
| 全部 4 bit | 0.96 | −0.003 | −0.017 |
| 全部 3 bit | 0.64 | −0.005 | −0.016 |
| 全部三值 | 0.32 | −0.036 | −0.048 |
| **FFN + B/C 三值，其余 8 bit** | **1.45** | **−0.012** | **−0.010** |
| in_zx + out 4 bit，其余三值 | 0.70 | −0.015 | −0.031 |
| in_zx 4 bit，其余三值 | 0.57 | −0.019 | −0.035 |

逐层组敏感度（其余层组 8 bit，只把一个层组改成三值），按 source 验证 R² 的变化：

- B/C 行：−0.001 到 +0.008，三值几乎不受影响。
- FFN 单层：−0.001 到 −0.009。
- in_zx（in_proj 中 z/x 部分的行）：M1 第 1 层 −0.016，M2 第 0 层 −0.020。
- out_proj：M2 第 1 层 −0.024。

结论：三值并非“几乎无损”。B/C 行和 FFN 适合用三值；in_zx 和 out_proj 应保留较高精度。以上全部是 PTQ 结果，配合量化感知训练应能缩小损失。

只评估、不重训。模型为 Mamba-3 128×2（state 16）的 source checkpoint（`results/debug_round2/latest/{m1,m2}/cross_session_mamba3_official_w128_l2_n16_ctx128_seed0`）。target 输入使用 support 段 z-score（只用神经数据），即审核文档中的 P0 合同。指标为 query all-valid 的直接 R²；“/ ridge” 表示加上 support 段输出 ridge 校准后的结果（O×O 的闭式解，片上可以计算）。

## 设定

- **放在 RRAM 的权重（每个 block）**：`ssm.in_proj` 的 z/x 行、B/C 行，`ssm.out_proj`，`ffn.0`，`ffn.2`，共 335,872 个。其余部分（根层投影、dt/A/trap/angle 行、各种小向量、递推状态）放在数字端，按浮点处理。
- **2T2R 编码**：一个差分对存一个有符号数位，取值 {−1, 0, +1}。
  - 三值（1.58 bit）：每个权重 1 对，阈值按 TWN 取 0.7·mean|w|，每个输出列一个数字尺度。
  - k bit：对称取整，按行搜索使均方误差最小的截断范围，再用符号–幅值的二进制位切片存储，每个权重 k−1 对，各切片在数字端移位相加。
- **激活**：RRAM 层的输入做 INT8 量化，静态尺度只在 support 窗口上标定。
- **器件模型**：低阻态（LRS）有相对波动 σ，开关比 20；差分对的系统性增益（1 − 1/ratio）假定已由每列的数字尺度吸收。每个 σ 取 3 个随机“芯片”。
- **没有建模**：ADC 和部分和量化、IR drop、电导弛豫、写入校验误差。

## 结果

浮点基线：M1 0.623 / 0.595，M2 0.223 / 0.247。

| 配置 | 2T2R 对数（Mi） | M1 | M1 σ=10%（3 芯片均值 [范围]） | M2 | M2 σ=10% |
|---|---:|---:|---:|---:|---:|
| 全部 8 bit | 2.24 | 0.623 / 0.594 | 0.619 [0.614, 0.622] / 0.591 | 0.228 / 0.251 | 0.221 [0.216, 0.224] / 0.251 |
| 全部 4 bit | 0.96 | 0.619 / 0.590 | 0.614 [0.612, 0.616] / 0.586 | 0.221 / 0.245 | 0.220 [0.213, 0.230] / 0.237 |
| in_zx 8 bit，其余三值 | 1.07 | 0.620 / 0.603 | 0.620 [0.618, 0.623] / 0.603 | 0.203 / 0.253 | 0.196 [0.187, 0.203] / 0.247 |
| in_zx 4 bit，其余三值 | 0.57 | 0.619 / 0.601 | 0.614 [0.611, 0.616] / 0.600 | 0.196 / 0.251 | 0.199 [0.190, 0.204] / 0.247 |
| FFN 三值，其余 8 bit | 1.49 | 0.618 / 0.604 | 0.614 [0.609, 0.619] / 0.600 | 0.214 / 0.237 | 0.209 [0.201, 0.217] / 0.239 |
| 全部三值 | 0.32 | 0.620 / 0.612 | 0.617 [0.614, 0.620] / 0.613 | 0.200 / 0.238 | 0.202 [0.188, 0.210] / 0.236 |

逐层组的 target R² 敏感度见 `sensitivity.json`。按顶部“更正”一节，这些 target R² 的差异主要是噪声：同样大小的随机扰动就能让 target R² 变化约 ±0.05。层组之间的比较应以 source 验证 R² 为准。

## SRAM 预算（128×2，不含指令存储）

| 项目 | M1 | M2 |
|---|---:|---:|
| 根层 in/out_proj、final_norm（INT8） | 10.4 KB | 12.6 KB |
| dt/A/trap/angle 行（INT8） | 4.0 KB | 4.0 KB |
| 小向量（INT16） | 4.2 KB | 4.2 KB |
| 递推状态 8,864 个（INT24） | 26.0 KB | 26.0 KB |
| LoRA rank-4 适配器（INT8） | 9.7 KB | 9.8 KB |
| 激活缓冲（INT16，双缓冲） | 9.4 KB | 9.4 KB |
| EMA 统计、计数器、输出 ridge | 2.7 KB | 0.9 KB |
| **合计** | **66.2 KB** | **66.8 KB** |

Mamba-3 256×4 光递推状态就有 68,864 个，INT24 下约 202 KB（INT16 下也有 135 KB），超过 128 KB 的 SRAM 约束。

## 局限

单个 checkpoint、单个 target。量化方式是 PTQ，三值如果配合量化感知训练（QAT），结果应该更好。ADC 和部分和量化没有建模，这是下一步最关键的风险。在这个扰动量级下，target R² 的噪声底约 ±0.05，比较量化方案应以 source 验证 R² 为准。

## 复现

```bash
cd /home/xinyuan/Work_host/SSM
env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/.tools/mamba_deps:$PWD TRITON_CACHE_DIR=/tmp/ssm_audit/.triton CUDA_VISIBLE_DEVICES=0 \
  /home/xinyuan/miniconda3/envs/spint/bin/python results/analysis/rram_quant_20261003/rram_ptq.py --phase sensitivity
env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/.tools/mamba_deps:$PWD TRITON_CACHE_DIR=/tmp/ssm_audit/.triton CUDA_VISIBLE_DEVICES=0 \
  /home/xinyuan/miniconda3/envs/spint/bin/python results/analysis/rram_quant_20261003/rram_ptq.py --phase mixed   # 配置见 mixed_configs.json
```
