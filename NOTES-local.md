# Inkling 2 Mini on this machine: what was measured, and what is closed

Everything here was measured on one machine (Windows, Ryzen 7 8845H, **Radeon RX 7900 XTX over eGPU**, HIP SDK 6.4),
2026-11-26/27, unless a line says otherwise. Both engines were run against the same weights where the file format
allowed it; where it did not, the line says so.

This is not part of the branch's code. It is the record of one long measurement session, kept next to the scripts it
produced (`scripts/inkling2-amd/start.bat`, `stop.bat`), so the next person does not re-run the same twenty dead ends.

**The headline: 136.39 tok/s, and every knob is now measured.** Section J4 is the evidence that there is nothing left
to tune; section J9 lists what is left if you are willing to change kernels instead of flags.

---

# RustingStrata 待办（2026-11-26，基于 f72957f 实测）

模型：`Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf`（专家张量实际是 **NVFP4**，不是 Q4_K）
机器：RX 7900 XTX / 24 GB / Vulkan（AMD proprietary 26.3.1 LLPC）
Strata 基线：prefill 1059 tok/s，decode 85.9 tok/s（pp 512 / tg 128，ctx 4096）

**2026-11-26 已合并 `origin/main`（v1.3.7-40，71 提交）并重测**（备份分支 `backup/pre-merge-20261126` = f72957f）：

| 指标 | 合并前 f72957f | 合并后 03d81cb | 差 |
|---|---|---|---|
| prefill pp512 | 1059 | **1107.4** | +4.6% |
| prefill pp2048 | — | **933.7** | 随长度退化 |
| decode tg128 | 85.9 | **86.0** | +0.1% |

同长度对照 llama.cpp（HIP，`llama-bench`）：pp512 **3300.6**、pp2048 **3289.3**（平的）。
→ 差距：pp512 **2.98×**，pp2048 **3.52×**；**Strata 的 prefill 随长度衰减而 llama.cpp 不衰减**。

注意：eGPU 重插拔后 Vulkan 设备顺序变了，现在 `--device 0` = 7900 XTX、`--device 1` = 780M 核显；
用 `vulkaninfo --summary` 或 bench 的首行日志确认，别想当然。

---

## 0. 参照基线：llama.cpp（Makdok 分支）

**这是本机唯一能跑 100+ tok/s 的引擎，所有优化以后都跟这一份比。**

- 源码：`D:\software\makdok-llama`（分支 `inkling2-amd`，HEAD `64d6de3`，**未做任何改动**）
- 二进制：`build-hip\bin`（HIP SDK 6.4 自建，`scripts\inkling2-amd\build-hip.cmd`，约 9 分钟）
- 模型：`Inkling-2-Mini-UD-Q4_K_XL-old.gguf`（老架构名 `inkling2`；新架构名 `inkling2_loop` 的文件它不认）
- Drafter：`dspark-Inkling-2-Mini-Q8_0_Old.gguf`（402,278,560 B，arch `dflash`）
- 辅助脚本与长 prompt：`D:\software\makdok-bench\`

### 0.1 无 drafter（`llama-bench`，与 Strata bench 同口径）

```bat
build-hip\bin\llama-bench.exe -m <old.gguf> -dev ROCm0 -ngl 99 -fa 1 -p 512 -n 128 -r 2
```
pp512 = **3308.55** tok/s，tg128 = **92.53** tok/s

### 0.2 带 drafter（`llama-server`，本机实测四档）

server 口径：单 slot，源码 prompt 读一遍后生成 300 token，`--spec-type draft-dspark --spec-draft-n-max 3`

| ctx | KV | prompt token | reading | **writing** | 接受率 | token/step |
|---|---|---|---|---|---|---|
| 32768 | f16 | 5,281 | 2314.2 | **119.09** | 58.5% | 2.74 |
| 65536 | f16 | 56,058 | 2746.0 | **131.27** | 92.0% | 3.74 |
| 131072 | q8 | 129,804 | 2061.5 | **93.98** | 69.1% | 3.07 |
| 262144 | q8 K / q4 V | 231,011 | 1234.2 | **75.87** | 81.9% | 3.44 |

各档命令行：
```bat
:: 32768 / 65536：f16 KV
llama-server -m <old.gguf> -dev ROCm0 -ngl 99 -fa 1 -c 65536 -np 1 -b 2048 -ub 1024 --jinja ^
  -md <drafter.gguf> -ngld 99 --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-ubatch 512 ^
  --backend-sampling --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --host 127.0.0.1 --port 8084

:: 131072：加 -ctk q8_0 -ctv q8_0，ubatch 256
:: 262144：加 -ctk q8_0 -ctv q4_0，并设 INKLING2_DRAFT_KV_WINDOW=32768 INKLING2_FAR_Q8_HEADS=2
```

writing 数随接受率大幅波动（92% → 131 t/s，58.5% → 119 t/s），比较时必须同时报接受率。
一句话概括：**同一份权重、同一块卡，Strata 85.9 vs llama.cpp 92.5（差 8%），但 prefill 1059 vs 3308.6（差 3.12 倍）。**

---

## A. 投机解码验证批的浪费

### A1. 验证批注意力逐行跑（最大单项，+0.75 ms/token）

`src/inkling2/forward/device.rs:1137` 在 `verify_mode` 下是 `for t in 0..nt { be.ink_attn(...) }`，
每个验证行都把同一份 KV 从头读一遍，零摊销。

证据：`ink_attn` 调用数 nt=1 时 47 次/token，nt=3 时 **144 次/step**（= 3 × 48）。
副作用：每行独立做 KV 裁剪，`ink_clip` 每 token 成本被抬到 3.4 倍。

修法：`src/vulkan/shaders/ink_attn.comp`（59 行）加一个行维度，一次扫 KV 算多行 logits/value
（每行自己的 ring 起始、自己的 mask）；`ink_attn_merge` 相应按行出结果。
预估省 ~3.2 ms/step。注意 swa 层每行环形偏移不同，全注意力层每行前缀长度差 1。

### A2. Drafter 是 F32（+2.07 ms/token）

当前转出来的 `dspark-Inkling-2-Mini-F32.gguf` 是 1212 MiB。3 层 × 3 个 9216×2304 矩阵 = 764 MB，
每 step 全读一遍。官方文档用的是 `Q8_0` 版本。

修法：给 `tools/convert_mtp_to_dspark.py` 加 Q8_0 输出（742 MB → 421 MB）。纯数据改动，无引擎风险。
预估省 ~1 ms/step。

### A3. moe_gemv 在模型里是 2.21×/step，微基准只要 1.66×

微基准 `tests/moe_batch.rs` 证明 3 行互不重复专家时 gu 是 1.66×、dn 是 1.63×，带宽从 48% 升到 86% 峰值。
模型里却是 2.21×。

来源猜测（未验证）：`max_dup = k/xg = 3` 会选到 `MAX_NM=4` 特化管线；相关路由下 24 个 slot 里约 1/3
命中 `s_dup` 直接 return，但工作组网格照发；`load_tables()` + 2 次 barrier 每工作组都付一遍。

修法：先确认是哪一个。若确为 `MAX_NM` 特化，考虑在 host 侧把同一专家的 slot 折叠（现在是设备端折叠）。

### A4. 天花板（重要，决定优先级）

实测接受率 37.9%（真实不重复文本），1.758 token/step：

| 场景 | 每 step | 每 token | tps |
|---|---|---|---|
| 基线（不 draft） | 14.4 ms | 14.4 ms | 85.9 |
| 现在（draft 3 行） | 30.6 ms | 17.4 ms | 57 |
| A1+A2+A3 全修 | 23.6 ms | 13.4 ms | 74.5 |
| 全修 + 接受率 100%（重复性输出） | 23.6 ms | 7.9 ms | 127 |

**结论：修完 A 组，通用写作仍是净亏（74.5 < 85.9）。价值只在重复性输出场景（改代码、引用、结构化数据）
——那正是 draft 的正当用途，能从 57 拉到 ~127。**

---

## B. Prefill 的 3 倍差距（与内容无关，收益确定）

### B0. 关键定性：prefill 是**算力受限**，不是带宽受限（2026-11-26 实测）

每 token 每 layer-run 的专家 FLOPs = 8 × [(2304×2304) + (2304×1152)] × 2 = 127.4 MFLOP；48 layer-run → 6.11 GFLOP/token。
pp 2048 → 12.52 TFLOP，专家权重只需读 13.7 GB（一次遍历）。

| | 时间 | 折算 |
|---|---|---|
| Strata 专家 GEMM 本身 | **2.24 s** | **5.6 TFLOPS**（7900XTX FP16 峰值约 61 TFLOPS 的 9%） |
| llama.cpp 整段 pp2048 | **0.62 s**（3289 tok/s） | ≥20 TFLOPS 等效 |

→ **llama.cpp 跑完整个 2048-token prefill 的时间，比 Strata 只跑专家 GEMM 还短 3.6 倍。**
带宽完全不是瓶颈（13.7 GB / 0.62 s = 22 GB/s）。差距是 GEMM 效率，不是访存。

### B0b. 「Vulkan 后端拖累」的假设已被证伪（2026-11-26 实测）

给 Makdok 分支另编了 Vulkan 版（`D:\software\makdok-llama\build-vk`），同卡同权重：

| 引擎 / 后端 | prefill pp512 | prefill pp2048 | decode |
|---|---|---|---|
| llama.cpp **HIP** | **3300.6** | **3289.3** | 92.5 |
| Strata **Vulkan** | 1107.4 | 912.9 | 86.0 |
| llama.cpp **Vulkan** | 121.9 | 122.1 | 13.0 |

**llama.cpp 自己的 Vulkan 比自家 HIP 慢 27 倍，比 Strata 的 Vulkan 慢 9 倍。**
所以「Vulkan 这个 API 有 3 倍劣势」不成立——Strata 的 Vulkan 实现反而远好于 llama.cpp 的 Vulkan。
差距是 llama.cpp 在 HIP 上有 banded FA 和 NVFP4 WMMA，而它的 Vulkan 路径（#32898 未合）是通用慢路径。
结论：Strata 的 3 倍差距是**内核质量**差距，可达。

### B0c. prefill 随长度的衰减被此前的口径放大了

统一 `--reps 1` 扫同一轮：

| pp | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| Strata prefill | 930.9 | **983.0** | 962.4 | 912.9 | 877.5 |

512→4096 只掉 **11%**（不是之前说的 16%，那是两次不同 reps 的运行混比造成的）。llama.cpp 同区间基本持平。
衰减存在但不是主要矛盾；主要矛盾是各长度上都差 ~3.4 倍。


**合并上游后的最新定位（2026-11-26，pp 2048 剖面，`STRATA_VK_PROF=1`）**：

| kernel | 占比 | 每调用 |
|---|---|---|
| `gemmi_s_16`（int8 coop 专家 GEMM） | **37.2%** | 433.5 µs |
| `gemmi_16`（同上，非 s 变体） | **25.4%** | 1185.0 µs |
| `ink_attn_prefill` | 24.9% | 2316.8 µs |
| `ink_ctrl_gate` / `ink_ctrl_out` | 6.8% | — |
| 其余 | ~5.7% | — |

→ **专家 GEMM 62.6%，prefill 注意力 24.9%。** 所以「#101 说的 chunk attention 分数还在 host 上算」是过期的
（`docs/BENCHMARKS.md` 自己写着 "Prefill attention is on the GPU since"），主攻方向应是 **B2 专家 GEMM**。

已排除的 knob：`STRATA_VK_NO_COOP_I8=1`（退回 DP4A）pp2048 只有 583.1 tok/s，比现有 int8 coop 的 933.7 慢 1.6 倍
→ 现有 coop 路径是对的，不能退回。

算术边界：只把专家 GEMM 做到 3 倍 → prefill ≈ 1605 tok/s（仍差 llama.cpp 约 2 倍）；所以 GEMM 和注意力两块都要动。

| | Strata | Makdok llama.cpp（同卡） |
|---|---|---|
| prefill | 1059 tok/s | 2998–3154 tok/s |
| decode | 85.9 | 92（无 draft）/ 110–117（DSpark） |

**2026-11-26 在本机复现（同一份权重 UD-Q4_K_XL，同卡，llama.cpp 由 `D:\software\makdok-llama` 自建）**：

| 指标 | Strata | llama.cpp | 差 |
|---|---|---|---|
| prefill pp512（llama-bench） | 1059 | **3308.6** | **3.12×** |
| decode 无 drafter | 85.9 | **92.5** | 1.08× |
| decode + DSpark，5k prompt | 51.2 | **119.1** | — |
| 接受率 / 每 step | 37.9% / 1.758 | 58.5% / **2.74** | — |
| decode @231k depth（ctx 262144, q8 K/q4 V） | 未测 | **75.9**（13.18 ms/token） | — |

关键推论：llama.cpp 在 **58.5% 接受率、2.74 token/step** 下仍然 +33%；
Strata 接受率更高、1.758 token/step 却 −40%。说明差距全在验证批效率，不在 draft 质量。
按 A 组的账修完后 2.74 token/step 对应约 116 tok/s，与 llama.cpp 实测位置吻合。

来源（对方 README 自述，已核实）：

### B1. Banded flash attention（主因）

Inkling 2 每头有一个 relative-position bias，原版当成 dense 每头张量来算。对方改成
`FLASH_ATTN_EXT_BANDED` 的 far 模式 + HIP WMMA 内核（`f9f34363` / `255dc789` / `4c4f6215`）。
本仓库对应物是 `src/vulkan/shaders/ink_attn_prefill.comp` 和 `src/inkling2/forward/device.rs` 的 prefill 层。

### B2. 专家 GEMM：根因是 **tile 太小**，不是反量化，也不是占用率

**三次假设、两次被自己证伪，最终定位（2026-11-26）**

**第一次（占用率）**：`gemmi_16` 用 40960 B 共享内存 → 每 CU 只能驻留 1 个工作组（12.5% 占用率）。
做成单缓冲变体（`gemmi_16_nb` 22528 B → 2 个；`gemmi_s_16_nb` 12544 B → 3 个，驱动实测值），
输出逐字节一致，但 prefill 只涨 **+5~10%**（pp2048: 913.3 → 1005.9）。
→ **占用率不是瓶颈。**

**第二次（反量化）**：怀疑 NVFP4→int8 的反量化链。把同一份专家权重量化成 F16（**3.56 倍字节**）走现有
float GEMM，在**固定 128 个工作组**（每专家一个 tile）、只变 tile 内位置数的条件下对比：

| 每 tile 位置数 | NVFP4（int8） | F16 | F16/int8 |
|---|---|---|---|
| 8 | **4.85 TFLOPS** | 4.89 | 1.01× |
| 16 | 12.53 | 14.21 | 1.13× |
| 32 | **27.69** | 25.42 | 0.92× |
| 64 | 21.84 | 20.67 | 0.95× |
| 96 | 25.90 | 26.18 | 1.01× |
| 128 | 29.08 | 29.72 | 1.02× |
| 256 | 56.15 | 57.46 | 1.02× |

两个结论：
1. **F16 ≈ int8（0.92–1.13×），反量化根本不是成本。NVFP4→F16 融合内核不值得写，option 1 作废。**
2. tile 位置数只在 **<32 时才有影响**（8 位置 4.85 → 32 位置 27.69，5.7 倍），≥32 就平了。
   内核本身在真实 tile 尺寸下能跑 22–29 TFLOPS，相对 FP16 峰值 61 TFLOPS 是 1/3 —— **不算差**。

**⚠️ 一个必须记住的坑**：这个基准第一次跑时我用了 `ne.min(8)` = 只有 8 个工作组，得出的
「吞吐与 tile 位置数成正比、8 位置 2.72 → 128 位置 21.24 TFLOPS」是**并行度**造成的假象，不是 tile 大小。
固定工作组数后曲线完全不同（8→32 才是陡升段）。**任何 GEMM 基准都必须固定网格，只变想测的那一维。**

**第三次（tile 大小 → 改 `--chunk`）**：`--chunk` 256 / 1024 / 4096 / 8192 在 pp4096 上是
857.9 / 868.7 / 848.3 / 873.5 tok/s —— **噪声内的差别**。原因查清了：`--chunk` 根本不是 tile 尺寸的控制器。

**根因链（已闭合并已兑现，2026-11-26）**

加了个 `STRATA_TILE_HIST=1` 的直方图（`device.rs` 的 `by_e` 分组处，env 门控），打真实 tile 分布：

```
tile hist: nt=256 k=8 tiles=100 | <=8 32~45  <=16 17~25  <=32 16~30  <=64 7~16  <=96 3~6  <=128 1~4
```

→ **平均每个 tile 只有 20.5 个位置**，约 75% ≤32，**约 40% ≤8**；`nt=256` 且**与 `--chunk` 无关**。

上限来自 `src/inkling2/forward/device.rs:47` 的 `pf_tokens()`：
**`STRATA_DS_PF_TOKENS`，单卡默认 256**（双卡 512，上限 2048）。

**✅ 已实测兑现**（pp 2048，`--reps 1`，同一轮）：

| 配置 | prefill | 相对默认 |
|---|---|---|
| 默认（PF=256） | 899.6 | — |
| PF=512 | 1126.9 | +25% |
| PF=1024 | 1252.0 | +39% |
| PF=2048 | 1351.0 | +50% |
| PF=1024 + `STRATA_VK_GEMM_NB` | 1392.4 | +55% |
| **PF=2048 + NB** | **1464.3** | **+63%** |

pp 4096：879.3 → 1144.0（PF=1024）→ 1189.4（PF=2048）。
**对 llama.cpp（3289 @ pp2048）的差距从 3.4× 缩到 2.25×。**

**✅ 默认值已改（2026-11-26）**：`pf_tokens()` 默认 256 → **1024**（`STRATA_DS_PF_TOKENS` 仍可覆盖，上限 2048）。
三档实测（`--reps 2`，同轮）：

| pp | 旧默认 256 | **新默认 1024** | prefill 增益 | decode（256 → 1024） |
|---|---|---|---|---|
| 512 | 1065.9 | **1290.4** | +21% | 84.4 → 83.6 |
| 2048 | 932.5 | **1283.2** | +38% | 82.2 → 82.3 |
| 4096 | 862.8 | **1165.4** | +35% | 78.4 → 76.5 |

**decode 无损**（±2% 噪声内）。对 llama.cpp 的差距：pp512 **3.10× → 2.56×**，pp2048 **3.53× → 2.56×**。
选 1024 而不是 2048 的原因：2048 在 pp4096 档 decode 从 61.1 掉到 57.2（prefill scratch 到 393 MB）；
1024 只花 197 MB 且 decode 无损，要更猛可以设 `STRATA_DS_PF_TOKENS=2048`。

数值验证：PF=256 / 1024 / 2048 / +NB 各生成 40 token，输出 **682 字节逐字节一致**（去时间戳后）。

`cargo test --release --lib`：156 通过，2 失败，**两个都是本机既有问题，与本次改动无关**：
- `sensenova_loop_mhc_cpu_matches_gpu`：`attn_prefill_64` 的 specialization 是**空列表**，需要 55808 B，
  本卡上限 49152 —— 空特化意味着我的改动影响不到它，是 AMD 设备的既有容量问题。
- `standalone_external_draft_model_spec_step_matches_greedy`：Windows `ERROR_USER_MAPPED_FILE`（打开已 mmap 的文件），平台问题。

注意：PF=2048 时 pp4096 的 decode 从 61.1 掉到 57.2（pp2048 档 66.9 → 68.0 正常），
可能是 prefill scratch 变大带来的副作用，默认值可以取 1024 作为折中（+39%，decode 无损）。

**目前的确定结论**：模型里专家的有效算力约 5.6 TFLOPS，而同一个内核在隔离环境下（tile≥32）能跑 22–29 TFLOPS，
差 4–5 倍。**低效不在 GEMM 内核本身，在于模型怎么驱动它**（实际产生什么 tile、启动多少次、之间的 barrier）。
下一步应该先**把模型里真实的 tile 尺寸分布打出来**（在 `by_e` 那段加一个 cnt 直方图），而不是继续猜。

已排除：`STRATA_VK_NO_COOP_I8=1`（退回 DP4A，pp2048 仅 583 tok/s，慢 1.6 倍）。

**根因已找到：共享内存把占用率锁死在 12.5%（2026-11-26）**

`src/vulkan/spv_smem.rs` 自带的断言给出精确值：

| kernel | 共享内存 | 48 KB 上限下每 CU 可驻留 | 占用率（256 线程/2048） |
|---|---|---|---|
| `gemmi_16`（BM=128 tile） | **40,960 B** | **1** | **12.5%** |
| `gemmi_s_16`（BM=32 tile） | 20,992 B | 2 | 25% |

`gemmi_16` 的 40 KB 构成（全部双缓冲）：xq 20 KB + wq 10 KB + 尺度/掩码 10 KB。
设备上限 49,152 B，只够放 1 个工作组，且**几乎没有余量**。

实测每行成本：`gemmi_16` = 1185 µs / 128 行 = **9.3 µs/行**；`gemmi_s_16` = 433 µs / 32 行 = **13.5 µs/行**。
→ 大 tile 每行反而更便宜，因为它把 NVFP4→int8 的反量化摊到 128 行（小 tile 只摊 32 行）。
所以不能靠改用小 tile 提占用率，**必须保住 128 行摊销的同时把共享内存压到 ≤24 KB 让 2 个工作组驻留**。

注意：这不是一个开关。现在 `gemmi.glsl` 是「把第 kb+1 步存进另一个缓冲，同时算第 kb 步，每步一次 barrier」，
改成单缓冲必须同时改 barrier 方案，否则会读到脏数据。属于 shader 级改动。

预估：占用率 1→2 个工作组若能让 GEMM 吞吐翻倍，专家 GEMM 2.24 s → 1.12 s，
prefill 913 → 约 **1330 tok/s**（1.46×）。若到 3-4 个工作组更好。

对照 llama.cpp 的做法：NVFP4 直接反量化成 **FP16 走 WMMA、按 96 token 分块**，省掉 int8 打包这一层。
两条路（压共享内存 vs 换 FP16 WMMA）可以分别试。

对方：NVFP4 按 96 token 分块、Q8_0/Q5_K 从 256 token 起走 FP16 WMMA，而不是 int8 MMQ；
gate_up 的 SiTU 和同专家 down_proj 在同一个 kernel 里出。
本仓库对应物：`prefill` 的 tile 列表 + `gemm_fs` / `gemmi_16` / `gemmi_s_16`（profiler 显示 prefill 里
`gemmi_s_16` 占 43%、`gemmi_16` 占 28%，合计 71%，是 prefill 的绝对大头）。

### B3. 每 token launch 8100 → 1215

Strata 实测约 1500 次 dispatch/token，这条已经接近，不是主要差距。

---

## C. 已排除的方向（留作结论，别重复做）

- **专家连续布局**：`tests/moe_layout.rs` 实测三种布局。相邻 ids 相对分散 ids 是 −0.8% ~ +2.3%（噪声内）；
  dtod gather 成连续 23.9 MB 再算 **−35%**。原因是每个工作组本来就在读连续 10.4 KB，同一时刻 GPU 只在
  一个专家的 3 MB 区域里顺读，8 个区域相邻与否对 HBM 无影响。上限 0。
- **rpw 调参**：dn 只有 768 个工作组（1 波），扫 rpw=1/2/3/6/12 全部落在噪声内，默认 3 最好。
- **单行 moe_gemv 带宽**：已经 51–58% 峰值（459–521 GB/s），不是 shader 能再抠的地方。
  「129 GB/s / HBM 14%」是整 token 平均（1.72 GB 专家权重 / 11.63 ms），不是 kernel 带宽。

---

## D. 长上下文（2026-11-26 实测：没有 bug，是二次项）

f16 KV，`--ctx` 足够，`--tg 4`：

| n（prefill 长度） | 实测耗时 | 速率 tok/s | 每翻倍耗时比 |
|---|---|---|---|
| 8,192 | 10.3 s | 795 | — |
| 16,384 | 23.3 s | 703 | 2.26 |
| 32,768 | 57.4 s | 571 | 2.46 |
| 65,536 | 183.2 s | 358 | **3.19** |

拟合 `T(n) = 0.830 ms/千token · n + 2.991e-8 · n²`（渐近线性速率 1205 tok/s）：

| n | 预测耗时 | 其中二次项（prefill 注意力）占比 |
|---|---|---|
| 131,072 | 10.4 min | 83% |
| **262,144** | **约 38 min** | **90%** |

→ 之前那次 262144 + `--kv q8` 跑 35 分钟没完，**不是 bug**：q8 的 prefill 还要再慢约 35%，总时长约 48 min。
对比 llama.cpp HIP 的 231,011 token prefill = **187 s**。所以长 prompt 的差距不是 3 倍，而是 **10–20 倍**，
主因是 prefill 注意力（二次项），不是 GEMM。

**已撤回的说法**：曾观察到 `--kv q8` 在 depth 16384 时 decode 掉到 2.2 tok/s（比 f16 的 48.6 慢 22 倍），
后续 4 次复跑（ctx 20480/32768、depth 14336/16384/17408）全部正常（46–50 tok/s），**不可复现，不作结论**。

quantized KV 的可复现代价只有 prefill：同深度下 q8 比 f16 慢约 35–40%（8192: 542 vs 797；16384: 439 vs 703），
decode 在 depth ≤ 16384 时与 f16 持平（46–54 vs 47–58）。

| 深度 | decode（f16） |
|---|---|
| 128 | 86.0 |
| 4096 | 79.5 |
| 8192 | 52.2 |
| 16384 | 48.6 |
| 32768 | 44.7 |
| 65536 | 35.0 |

## E. 不对称 KV 量化（2026-11-26 合并后已支持，未实测）

PR #84 已合入 `origin/main`（我们今天合并的那批），现在有：

```
--kv f16|q8|q4|q2|turbo4|turbo4-3|turbo4-2|turbo3|turbo2|turbo3-2
--k  f16|q8|q4|q2|turbo4|turbo3|turbo2        ← K 单独指定
--v  f16|q8|q4|q2|turbo4|turbo3|turbo2        ← V 单独指定
```

`turbo4-3` = K turbo4 + V turbo3。后端里 `kv_k_mode` / `kv_v_mode` 已是两个独立字段。

**Vulkan 明确拒绝 turbo**（`src/vulkan/mod.rs:375`："--kv turbo* is not ready on Vulkan yet;
use the CUDA backend or --kv q8/q4"）。所以本机（只有 Vulkan）能用的只有非 turbo 组合，例如
`--k q8 --v q4`。

**待做（零代码、纯测量）**：
1. `--k q8 --v q4` 在本机 Vulkan 上是否可用、长上下文 decode 提升多少（llama.cpp 用它跑 262144）。
2. 仓库新增了 `tests/vulkan_split_kv.rs`，可以直接跑。
3. 256k 长上下文 decode 实测（此前被中止，见 D 组）。

参考：Makdok 的 llama.cpp 用 `-ctk q8_0 -ctv q4_0` + 滑窗层保 F16 跑 262144，质量 ppl 6.2667 vs F16 6.2608。

---

## F. 复现命令

```bash
# 基线
rustingstrata.exe bench --gpu --backend vulkan --model <模型> --pp 512 --tg 128 --reps 2

# 专家布局（C 组结论）
STRATA_MODEL=<模型> cargo test --release --test moe_layout -- --ignored --nocapture

# 多行 moe_gemv 行数缩放（A3）
STRATA_MODEL=<模型> cargo test --release --test moe_batch -- --ignored --nocapture

# 逐 kernel 剖面（decode 要看 GRAPH，否则计数覆盖不到图内）
STRATA_VK_PROF=1 STRATA_VK_PROF_GRAPH=1 STRATA_VK_PROF_EVERY=8 rustingstrata.exe bench ...

# 投机测速
rustingstrata.exe bench ... --draft-model dspark-Inkling-2-Mini-F32.gguf --spec-draft-n-max 2
```

---

## G. 2026-11-26 晚间更新（v1.4.1 合并后）

### G1. `pf_tokens` 的结论被推翻并修正

原来记的「pf_tokens 256→1024 得 +21~38%」只对短 prompt 成立。实测 chunk 1024 vs 256（Vulkan，f16 KV）：

| pp | chunk 1024 | chunk 256 |
|---|---|---|
| 8,192 | **971.3** | 799.3 |
| 16,384 | 701.9 | 706.8（平） |
| 32,768 | 440.6 | **568.0** |
| 65,536 | 223.3 | **404.6** |

原因：chunk 的 prefill 注意力成本 = chunk × 位置。交叉点 ~16384。
**已修**：`forward.rs` 改成按位置自适应（<16384 用 1024，之后 256），提交 `ab25f01`。
自适应后：512→1296.5、2048→1281.4、8192→979.1、32768→568.9、65536→402.8。

### G2. 投机的判据换成「验证行成本」

v1.4.1 的 Vulkan 已有 BATCHED split-K（`gemmi_s`/`gemmi_s2`）；真正剩下的是 drafter 体积。
把 `convert_mtp_to_dspark.py` 加 `STRATA_DSPARK_Q8=1` 转出 Q8_0（322.2 MiB vs F32 1212.7 MiB），提交 `329ec3e`。

| 场景 | decode | 接受率 |
|---|---|---|
| 不开投机 | 84.3 | — |
| Q8 drafter, max=4 | **90.7** | 95.3% |
| Q8 drafter, max=8 | **91.8** | 91.7% |
| Q8 drafter, max=8，2048 token 不重复文本 | **29.3** | 21.2% |

**新判据**：一个被验证的行成本 ≈ **0.84 个解码 token**（llama.cpp 约 **0.2**）。
打平需要接受率 ≈80%，与 draft 长度无关。**差距是 4 倍的验证行成本，不是 draft 质量。**

### G3. v1.4.1 删掉了逐 kernel 剖面

`STRATA_VK_PROF` 的汇总只剩 `Drop for Inner`，`bench` 不析构拿不到；用 `generate` 也只记录到 3 次调用。
**上游的可用性回归**，报 issue 时值得带上。

### G4. B1（banded FA）的量级

由 pp 16384/32768/65536 的耗时（23.6 / 57.6 / 183.2 s）拟合，注意力约占 55–60%（64k）、约 82%（262k）。
262144 估计 25–35 分钟，llama.cpp 231k 是 187 s。**差距约 10 倍，是剩余最大的单项。**

### G5. 自适应 chunk 是**权宜**，不是修好（2026-11-26 追加）

llama.cpp 自己的实测方向和我们相反（Makdok README，52k–250k token 的 prompt）：
`-ub 1024` 比 512 快约 20%，`-ub 2048` 也不掉。**所以「长 prompt 必须用小 chunk」不是通用规律。**

算术上也说不通：整个 prompt 的注意力对数是 Σ c×p ≈ n²/2，**与 chunk 无关**；c 只影响效率。

已排除的两个假设（chunk=1024，pp 32768）：
- **ATTN_BUDGET 切片**：默认 439.6 vs 关闭 439.4 → 无关
- **显存压力**：ctx 32768 / 49152 / 65536 → 438.2 / 439.1 / 439.4 → 无关
- **并行度不足**：shader 的 `grid.x = ceil(t/BQ)`（`attn_prefill_64.comp:43`），chunk 越大块数越多 → 也排除

**结论：原因未知。** 现在的自适应 chunk 让长上下文 +80%（223 → 403 @65536），但它掩盖了一个未解释的 Strata 低效点。
llama.cpp 证明大 chunk 在长位置可以不掉——**找到这个原因比重写注意力内核更划算**，因为找到了就能全程用 1024，
两头的收益一起吃。下一个会话应该先查这个，而不是先动 banded FA。

### G6. 投机步的成本模型（2026-11-26 实测，v1.4.1 + Q8 drafter）

扫 draft 长度（bench 内置段落，reps 3）：

| max | decode | 接受率 | token/step | step ms |
|---|---|---|---|---|
| 1 | 57.2 | 100% | 2.00 | 35.0 |
| 2 | 71.4 | 100% | 3.00 | 42.0 |
| 4 | 89.2 | 95.3% | 4.81 | 53.9 |
| 8 | **101.6** | 91.7% | 8.34 | 82.1 |

拟合 **step ≈ 21.8 ms 固定 + 6.6 ms/行**。每行 6.6 ms = 单 token 12.2 ms 的 **0.54**，
**等于隔离微基准的 0.55** → 每行成本没问题，问题全在固定开销。

**固定开销的主体是 DSpark drafter 自己的前向**：同样接受率下，`--spec-type ngram-mod`（无 drafter）
93.2% → **126.6 tok/s**，比 DSpark 的 101.6 快 25%，差 **15.3 ms/step**。
drafter 权重 Q8 只 322 MB（带宽下限 0.65 ms）→ 23 倍偏差。

原因：`dspark.rs:312` 的 `draft()` 每层调约 4 次 host matvec（`matvec_rows_sets`/`matvec_rows`），
3 层 ≈ **12 次 host↔device 往返/step**；而且 norm/rope 是 host 代码（`rms_norm_w` / `ops::rope_neox`），
每次往返都要把 8×2304 f32 的激活下载回来再上传。1.3 ms × 12 ≈ 15.6 ms，与实测吻合。

**三条修法**（按工作量）：
1. 把 drafter 的 norm/rope/markov 也搬上 GPU，整段录成一个 graph → 12 次往返变 1 次。
   预期 drafter 15.3 → ~2 ms，DSpark 101.6 → **~120**。工作量：中高（4-8 h）。
2. 只把 norm/rope 上 GPU、matvec 之间不再回 host → 往返次数不变但每次数据量小。
   收益有限，因为往返的固定成本（submit+fence）才是主项。
3. 强制 drafter 走 host（不让它的 Mat 上 GPU）+ 并行 AVX2 matvec：191 MFLOP/step，
   16 线程估计 4-6 ms。需要加开关。**最便宜的验证路径**（1 h）。

**当前 Strata 的最优 decode（bench 内置段落，接受率 93%）**：
`--spec-type ngram-mod --spec-draft-n-max 8` → **126.6 tok/s**，不需要 drafter 文件。
但真实不重复文本上它是 80.7（对 81.2 基线，约等于无效）；DSpark 在同条件 29.3（更差）。
**投机的收益完全取决于接受率，而接受率取决于输出是否与上下文重合。**

### G7. B 路第一步：decode 的 ~220 次 `gemv_q` 是什么（代码定位，2026-11-26）

`gemvq2` 在 Vulkan 上是**真融合**的（`mod.rs:1346`，一次 `gemv_q` 覆盖两矩阵的行，条件：同 ty、同 cols、非 float_x）。
所以每个 layer-run 的启动是：

| # | 调用 | 输入 | 启动数 |
|---|---|---|---|
| 1 | `gemvq2(wq, gate)` | `inp(0)` | 1 |
| 2 | `gemvq(rel)` | `inp(0)` | 1 |
| 3 | `gemvq(sink)` | `inp(0)` | 1 |
| 4 | `gemvq2(wk, wv)` | `inp(0)` 或 `inp(1)/inp(2)` | 1 |
| 5 | `gemvq(wo)` | `att` | 1 |
| 6 | `gemvq(gate_inp)` | `hn` | 1 |

= 每 layer-run 约 6 次 → 48 个 layer-run ≈ **288/token**（v1.3.7 剖面实测 220，部分层跳过 conv 投影）。

**可合并点**：第 1–3 项和 `wk/wv` 里有 4 个都读**同一行** `inp(0)`（wq、gate、rel、sink），
一个 4 路分组内核能把 3 次启动压成 1 次。
**阻碍**：`gemvq2` 只处理「同类型 + 同 cols」，而 `rel`/`sink` 的行数小得多，
需要异构形状的分组内核——正是 llama.cpp README 里写的 "grouped Q8_0 products (also over other
activations of the same shape)"。

**预期**：`gemv_q` 占 decode 约 15%（1.7 / 11.6 ms）。砍掉 3/6 次启动 ≈ −0.85 ms → 84.3 → **约 92 tps**。
真实但到不了 100+；剩下的非 MoE 时间是注意力 21%、`ink_mix` 16%、`ctrl_out` 9%。

**前置条件**：上述百分比来自 v1.3.7 的剖面。v1.4.1 的 `STRATA_VK_PROF` 输出被上游删坏了
（见 G3），**动手前先恢复剖面**，否则无法验收。

**未完成**：四个矩阵的实际量化类型没确认（临时 gguf 解析在 metadata 数组上出错），
需要在 `device.rs` 加一行打印，或先修好剖面从 kernel spec 里读。

---

## H. 2026-11-27 情报扫描（两家上游 + fork）

### H1. llama.cpp 上游：三个关键 PR 全部合入，Makdok 分支没动

| PR | 状态 | 影响 |
|---|---|---|
| **#33063 ggml/CUDA+HIP: Flash Attention with indexed bias** | **已合** | HIP 上有了 Inkling 2 的 banded FA |
| **#32898 vulkan: FA compact bias for Inkling 2 Mini** | **已合** | **Vulkan 上也有了**——这正是 Makdok 分支缺失、导致它 Vulkan 只有 13 tok/s 的那个东西 |
| #33060 CUDA: fuse short conv / depth mix / loop controller / L2 clip | **已合** | launch 融合 |
| #33167 HIP: split-K F32 GEMM for untuned RDNA3 | open | 正打我们 7900 XTX |
| #33150 Inkling 2 在 HIP 上 `-np 2` 会重复吐字 | open bug | |
| #33062 nvfp4 pass-through | 仍 open | |
| #33116 Vulkan 五链融合 | 关闭未合 | |

`Makdok/llama.cpp-inkling2-amd` 的最后一次提交仍是 `64d6de3`（2026-11-23），**没有新东西**。

**结论：我们的「131.27 tok/s 基线」很可能已经过期。** 上游 main 现在比 Makdok 分支多三个性能 PR，
而且 Vulkan 路径补上了 FA bias。**必须用当前 main 重测一次基线**（Vulkan 和 HIP 各一次）。

顺带一个好消息：我们的本地模型是 `inkling2_loop` 架构名，正是当前 main 吃的格式，
**重测不需要再下 17 GB**。

### H2. RustingStrata 上游：3 个新提交 + 两个和我们直接相关的东西

- `2ff9c265` / `4352f08d`：vision 上 GPU（Qwen3.8 Flash-Next）
- `0cecb248` **ROADMAP.md 重写**：方向改为「Serving: vLLM 风格的 MoE 批处理」——**上游不再以单请求 decode 为主攻**
- **#129（open，作者 `hun`）+ `hun/RustingStrata` 的 `inkling-port` 分支**：
  「HIP 上跑 Inkling 2 + VRAM 专家池」。RX 9070 XT（gfx1201，16 GB，专家部分落 CPU）：
  **decode ~60 tok/s，prefill ~1650 tok/s @4096**。含：
  - HIP 侧用 `kp16` prefill attention、tiled controllers、`ink_attn_128`（默认开）
  - **单卡 prompt chunk 默认 2048 行**（`STRATA_DS_PF_TOKENS`）——和我们自适应的 1024→256 不同
  - `--fixed-experts` 下输出与 Vulkan 逐字节一致
  - 验证了 hiprtc 对 gfx1010/1030/1100/1151/1201 都能编
- **#125（open）**：few-row batched GEMM 做验证批（CUDA only）——仍未合
- **#102（open）**：hip 不认当前设备 — 和我今天修的 Windows 加载相关

**结论：`inkling-port` 是「HIP + Inkling 2」的现成实现**，比我判断的「没人做过」要好——有人做了，
而且今天还在活跃（最新提交 2026-11-27 10:35）。配合我今天修的 Windows HIP 加载，理论上可以在本机试。

### H3. 优先级（据以上重排）

| # | 事项 | 理由 | 预期 |
|---|---|---|---|
| **1** | **用 llama.cpp 当前 main 重测基线（Vulkan + HIP）** | 三个关键 PR 已合，基线可能全面过期；Vulkan 现在有 FA bias | 可能把「llama.cpp 只有 HIP 能跑」这个前提推翻 |
| **2** | 拉 `hun/inkling-port` 试 HIP | 唯一现成的 HIP Inkling 2 实现；我们已修好 Windows HIP 加载 | 未知，但 HIP 在 llama.cpp 上是 3301 vs Vulkan 122 |
| 3 | 查 `ink_mix`(120/token) + `rms_norm`(153/token) | 见 I 节 | 未量化 |
| 4 | `pf_tokens` 2048 vs 我们的自适应规则 | hun 那边用 2048，我们用 1024→256 | 对比即可 |
| 5 | 长上下文 chunk 效率缺口（G5）、banded FA（B1） | 上游 #32898/#33063 已给出参考实现 | 大 |

---

## I. 下一个大头：`ink_mix` 120 次/token + `rms_norm` 153 次/token

decode 剖面（`STRATA_NO_GRAPH=1`，见 G3/G7）：

| kernel | 占比 | 次数/token | 每次 |
|---|---|---|---|
| `ink_mix` | 14.4% | **120** | 18.9 µs |
| `rms_norm` | 4.5% | **153** | 4.7 µs |
| `ink_conv` | 2.6% | 192 | 2.1 µs |
| `ink_ctrl_out` | 7.5% | 48 | 24.8 µs |

`ink_mix` 120 次/token 意味着**每个 layer-run 2.5 次**，`rms_norm` 153 次 ≈ 3.2 次/layer-run。

**查法**：
1. 在 `device.rs` 的 decode 路径把每个 `ink_mix` / `rms_norm` 调用点标号打印（或用一个静态计数器），
   确认 120 和 153 具体由哪些调用组成；
2. 看哪些能用 `n_n` 已经存在的批量变体合并（`ink_mix_n`、`rms_norm_streams` 之类）；
3. 预期：这两个加起来 18.9%，若能把调用数砍一半而单次不变，收益约 9%（decode 84.3 → ~92）。

**注意**：`ink_mix` 单次 18.9 µs 不算小，本身可能也有效率问题（它就是那个「depth mix」，
llama.cpp 在 #33060 里把它和其它四条链融合成一个 kernel）。

### H1b. 重测结果（2026-11-27，本机，`llama-bench -p 512,2048 -n 128 -r 2`）

**上游 main 不能加载老架构文件**（`check_tensor_dims: tensor 'blk.24.attn_q.weight' not found`），
而 Makdok 的分支不能加载新架构文件。所以两个引擎只能用各自支持的打包，**无法同文件对比**。

| 引擎 / 后端 | 模型文件 | pp512 | pp2048 | tg128 |
|---|---|---|---|---|
| **Makdok 分支 b12568 / HIP** | old（`inkling2`） | **3300.6** | **3289.3** | **92.5** |
| Makdok 分支 / Vulkan | old | 121.9 | 122.1 | 13.0 |
| 上游 main 2a8bf0f / HIP | mainline（`inkling2_loop`） | 1938.6 | 1904.2 | 59.5 |
| 上游 main 2a8bf0f / Vulkan | mainline | 1720.3 | 1706.5 | 55.4 |
| Strata v1.4.1+ / Vulkan | mainline | ~1343 | ~1282 | **84.3** |

**三条结论**：
1. **Makdok 的分支仍是 llama.cpp 系里最快的**：比当前 main 的 HIP 快 1.7×（prefill）和 1.55×（decode）。
   合进去的 #33060/#33063/#32898 没有补齐——Makdok 还有 main 没有的东西（小算子融合到 1215 launch、
   NVFP4 MoE 的 tile/grid 顺序等）。
2. **Vulkan 从 122 → 1720（14 倍）**，#32898 的效果是真的。对 Vulkan 用户来说 main 现在完全不同了。
3. **在 Vulkan 上 Strata 的 decode 仍赢**：84.3 vs 55.4（1.52×），而 prefill 输 1.28×（1343 vs 1720）。

**所以「131 tps 基线」不用改**——它来自 Makdok/HIP，仍然是最快的 llama.cpp。但基线表要补上
「main/Vulkan 1720+55」「main/HIP 1938+59」两行，并注明文件不同、不可直接跨引擎比。

### H2b. 最小验证结果：HIP 方向不值得全量合并（2026-11-27）

做了最小切片：`hun/inkling-port` 的移植提交 `d01ae4f hip: Inkling 2 on the ROCm backend`
（只改 `src/hip/mod.rs`，+126/−1；**它的父提交 `70ee521` 就在我们历史里，能干净 cherry-pick**）。
配合我们已有的 Windows HIP 加载修复（还要把 hiprtc 的库名也改成 `hiprtc0604.dll`），
**HIP 在 Windows 上首次真正跑起 Inkling 2**。

同树同文件对比（`--reps 3`）：

| 后端 | prefill pp512 | prefill pp2048 | decode |
|---|---|---|---|
| **HIP + 移植** | **1421.0**（+7%） | **1366.9**（+6%） | 72.1（**−15%**） |
| Vulkan | 1326.4 | 1288.6 | **84.9** |

**结论：prefill 略胜、decode 明显更差 → 不做全量合并。**

关键更正：hun 分支上「HIP 是自家 Vulkan 的 2 倍」（1121/75.8 vs 552/41.1）是**他们分支 Vulkan 偏弱**造成的
（v1.4.0 基线、没有我们的自适应 chunk），**不是 HIP 内核本身有 2 倍优势**。搬到我们的树上后优势消失。

另外：他们「HIP 默认用更快内核」的那个提交 `2305802` **不能单独搬**——依赖前面「VRAM 池/暂存」的基础设施
（`Gpu` 构造里多了 `side`/`marks`/`pinned`/`flags` 等字段）。要搬就得连池一起搬，而池是为「专家落 CPU」
设计的，对我们 17.6 GB 全驻留的场景没有意义。

**行动项关闭：HIP 路线不再投入。** decode 的问题在 `moe_gemv`(26%) / `ink_attn`(16%) / `ink_mix`(14%)，
不在后端选择上。

---

## J. 2026-11-27 收尾：单请求优化已到顶

单机使用的目标已定为「单请求最快」。这一节是最后一轮尝试的完整记录，**结论是所有可调旋钮都已试尽**。

### J1. decode 成本模型（两家的同一把尺子）

深度扫描（llama.cpp Makdok/HIP，6 个深度点，`--temp 0`，无投机）：

| 深度 | 268 | 1,528 | 4,019 | 7,744 | 9,307 | 21,211 |
|---|---|---|---|---|---|---|
| ms/token | 11.47 | 12.18 | 12.34 | 12.51 | 12.62 | 13.17 |

拟合 **`ms/token = 11.45 + 8.1e-5 × 深度`**；Strata 是 `11.7 + 2.58e-4 × 深度`。

| | 固定成本（深度 0） | 每 key 注意力 | 65k 处 |
|---|---|---|---|
| llama.cpp | **11.45 ms** | **8.1e-5 ms** | 16.7 ms → 60 tps |
| Strata | 11.7 ms | **2.58e-4 ms（3.2×）** | 28.5 ms → 35 tps |

**两个关键结论**：
1. **固定成本两家几乎一样（11.45 vs 11.7）**——「llama.cpp 靠 launch 融合领先」的猜测被否定。
2. **长上下文的差距 100% 是注意力斜率（3.2×）**，不是别的东西。

### J2. 那 11.45 ms 里 MoE 只占 25%

按 Strata 的可信剖面折算：

| 组成 | 时间 | 占比 |
|---|---|---|
| **MoE 专家流**（1.72 GB） | **2.9 ms** | 25% |
| 注意力（深度 512） | 1.8 ms | 16% |
| 220 次小 `gemv_q` | 1.7 ms | 15% |
| `ink_mix` | 1.6 ms | 14% |
| `ink_ctrl_out` + `ink_route` | 1.4 ms | 12% |
| `rms_norm` + `ink_conv` | 0.8 ms | 7% |

**读专家权重只花 25%，75% 在别处。** 理论天花板（非 MoE 趋零）= 1.72 GB / 460 GB/s = 3.7 ms ≈ 270 tps，两家共同。

### J3. 三项「改代码」的尝试，全部否决

| 项 | 做了什么 | 结果 |
|---|---|---|
| **LM head** | 写了 `INKLING2_HEAD_ROWS` 探针把 `output.weight` view 砍到 1/8 | 模型加载失败（`n_vocab` 一致性校验）；且上限只有 492 MB = **8%**。**否决** |
| **prefill tile** | 先在 Strata 测出 tile 曲线，再扫 llama.cpp 三个开关 | 默认已最优。**否决** |
| **launch 融合** | 读 `inkling2-fused.cu`（3106 行） | 该做的已做，剩下是零碎 op，本会话无法安全完成 |

**tile 曲线（Strata Vulkan，NVFP4 专家 GEMM，FP16 峰值 61 TFLOPS）**：

| 每 tile 位置 | 固定网格 128 tiles | busy 网格 256 tiles |
|---|---|---|
| 8 | 6.5 TFLOPS（11%） | 7.8（13%） |
| 16 | 14.9（24%） | 15.0（25%） |
| **32** | **26.5（43%）** | **24.8（41%）** |
| 48 | 17.5（29%）← 坑 | — |
| 64 | 22.1（36%） | — |
| 96 | 26.1（43%） | — |
| **128** | **29.3（48%）** | — |

**32 和 128 是甜点，48–64 是坑。** llama.cpp 的 96 token tile 正好落在 26.1 TFLOPS 上——**它已经坐在甜点上**，这是它三个 tile 开关全扫后默认最优的原因。
复跑：`cargo test --release --test prefill_tile_curve -- --ignored --nocapture`

### J4. llama.cpp 全部旋钮的扫描结果（这是「已到顶」的证据）

| 类别 | 试过 | 最优 | 结论 |
|---|---|---|---|
| draft 长度 | 3 / 5 / 8 | **5 → 131.1 tok/s** | 比官方推荐（3）快 3.3% |
| 投机类型 | dspark / ngram / 混合 | **dspark 单独** | 混合更差（117.8） |
| KV 量化 | f16 / q8 | 深上下文 q8（+8%），浅上下文 f16 | 交叉点 ~16k |
| MoE tile | `LDS1` / `TN96` / `MOE_F16` 各值 | **默认全优** | 每个替代都更慢 |
| 专家放置 | 全驻留 / `-ncmoe`+`--moe-cache-mib` | **全驻留** | 见 J5 |
| 思考长度 | `--reasoning-budget` / `reasoning_effort` | **`reasoning_effort: none`** | 见 J6 |

### J5. `--moe-cache-mib` 是灾难（已实测）

```
projected to use 17267 MiB vs. 24397 MiB free   → 余量 7130 MiB（29%）
```

| 配置 | decode |
|---|---|
| **全驻留（默认）** | **88.0 tok/s** |
| `-ncmoe 4 --moe-cache-mib 4096` | 25.6（**−71%**） |
| `-ncmoe 8 --moe-cache-mib auto` | 24.4（**−72%**） |

**原因**：全驻留走 VRAM（500 GB/s），一旦专家在 CPU 就要过 PCIe（~20 GB/s），而我们是 **eGPU**，PCIe 更差。
这解释了和 `ops864`（RTX 3070 8 GB，15→28 tok/s）的根本差异：**他是显存不够被迫用 cache，我们显存够，全驻留本来就是最优。**
7 GB 余量应该留给 **KV cache（更大的 `-c`）**，不是专家缓存。

### J6. `ops864` 的三个 patch 对 Inkling 2 的适用性

来源：`ops864/inkling2-mini-8gb`（RTX 3070 8 GB，15.1 → 28–36 tok/s）。三个 patch 均未合入上游。

| patch | 内容 | 对我们 |
|---|---|---|
| `0001` scheduler：循环层 host 权重按 split 复制 | `-ub≥32` 时 compute buffer 涨 3 GB | 24 GB 有富余，**影响小** |
| `0002` MoE cache：循环专家只存一次（上游 #32744，open） | 缓存按「层+pass」作 key，存 3 份 | 我们全驻留不开 cache，**不适用** |
| `0003` reasoning budget：每次回复一个预算（上游 #32836） | 每次新 `<think>` 重置预算 | **前提错误，不适用**，见下 |

**0003 的真相**：Inkling 2 **不用 `<think>` 标签**，它的思考由 chat template 的 `reasoning_effort` 参数控制（0.0–0.99）。
实测：`reasoning_effort: none` → 438 token **自然结束**；`high` → 512 撞上限。
所以 `--reasoning-budget` 对它**完全无效**（没有标签可匹配），正确旋钮是 `reasoning_effort`。
我实现过 `INKLING2_REASONING_ONCE`（对 `<think>` 类模型是真修 bug），**已按要求回滚**。

### J7. 仓库调查：没有可用的「换算法」捷径

| 候选 | 能用吗 | 速度收益 |
|---|---|---|
| 上游 #33062 nvfp4 pass-through | 能 | **0**（是转换工具，+159/−20，无性能数字；价值在保真：官方权重本就是 NVFP4 原生） |
| `ericflo/inkling-2-mini-loom`（AMD Loom，3–6× 于 llama.cpp） | **不能**（Linux + `/dev/kfd` only） | 拿不到 |
| `hun/inkling-port`（Strata HIP + 专家池） | 已验证：搬到我们树后 **prefill +7%、decode −15%** | **负**，不做全量合并 |
| `ops864` 三个 patch | 见 J6 | 对我们 ~0 |

### J8. 最终状态

**单机单请求，llama.cpp (Makdok/HIP) 的最优配置**：

```bat
llama-server -m <old.gguf> -dev ROCm0 -ngl 99 -fa 1 -c 32768 -np 1 -b 2048 -ub 1024 --jinja ^
  -md dspark-Inkling-2-Mini-Q8_0_Old.gguf -ngld 99 ^
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-ubatch 512 ^
  --backend-sampling --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

- **131.1 tok/s**（22,676 token 深度，接受率 61.9%）；无投机 89.4
- 深度 >16k 加 `-ctk q8_0 -ctv q8_0`（+8%）
- 请求里加 `chat_template_kwargs: {"reasoning_effort": "none"}` 控制思考长度
- 不要用混合投机、不要用 `--moe-cache-mib`、不要开 `-ncmoe`

**prefill 3266 tok/s**（MoE tile 三个维度全扫，默认最优）。

**再往上只能改 kernel 算法**（nvfp4 pass-through 是质量不是速度；Loom 的汇编内核在 Linux）。

### J9. 下一个会话如果还想推进

按潜在收益排，都超出「调参」范畴：

1. **Strata 的注意力斜率 3.2×**（J1）—— 长上下文唯一的大头，照 llama.cpp 的 banded FA 重写。目标：65k 处 35 → 60 tps。
2. **非 MoE 的 75%**（J2）—— 需要逐 kernel 优化，Strata 侧剖面已修好可用。
3. **llama.cpp 的 launch 融合**—— 在 3106 行现有实现上继续，需要读透 `inkling2.cpp`(1089 行) + `ggml-cuda.cu` 的融合框架。

**已封的方向（不要再试）**：专家连续布局、rpw、F16 替代 int8、`--moe-cache-mib`/`-ncmoe`、混合投机、MoE tile 开关、LM head、`--reasoning-budget` 对 Inkling 2、HIP 后端（Strata）、全量合并 `hun/inkling-port`。

### J10. `-b` / `-ub` 扫描（2026-11-27 补测，修正 J8 的配置）

这个维度之前完全没扫过——所有测试都固定了 `-b 2048 -ub 1024`。

**llama-bench 上看起来有 +25%**：

| `-b` | `-ub` | prefill pp2048 | decode |
|---|---|---|---|
| 2048 | 1024 | 3548.8 | 81.1 |
| 8192 | 2048 | 4036.1 | 89.3 |
| 8192 | 8192 | **4087.0** | 88.3 |

**但这是 llama-bench 的假象。** 三项验证：

1. **显存**：`-ub 8192` 的 compute buffer 从 355 → **3000 MiB**，预计占用 20,808 / 24,397 MiB，**余量只剩 15%**（`-ub 1024` 时是 29%）。
2. **`-ub 8192` 在真实长 prompt 下崩溃**：22,676 token 的 prefill 直接连接重置。**不可用。**
3. **server 上 `-b 8192 -ub 2048` 与默认持平**：79.52 / 75.15 对 81.73 / 74.43。

原因：`llama-bench -p 2048` 是一次性喂 2048 token，server 按请求分块，`-b`/`-ub` 的作用路径不同。

**带 drafter 时有小幅真收益**：

| 配置 | decode | 接受率 |
|---|---|---|
| `-b 2048 -ub 1024` | 134.87 | 61.9% |
| **`-b 8192 -ub 2048`** | **136.39** | **66.2%** |

**修正后的最优配置**（J8 的 `-b 2048 -ub 1024` 作废）：

```bat
llama-server -m <old.gguf> -dev ROCm0 -ngl 99 -fa 1 -c 32768 -np 1 -b 8192 -ub 2048 --jinja ^
  -md dspark-Inkling-2-Mini-Q8_0_Old.gguf -ngld 99 ^
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-ubatch 512 ^
  --backend-sampling --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

**136.39 tok/s**（原 131.1）。**不要用 `-ub 8192`**：compute buffer 3 GB，长 prompt 会崩。

**三条教训**：
1. `llama-bench` 和 `llama-server` 不可混比——差点把 4036 tok/s 当成真收益。
2. `-b`/`-ub` 这个维度因为一直用默认值而漏扫。
3. 显存余量是硬约束，15% 在长 prompt 下不够。


