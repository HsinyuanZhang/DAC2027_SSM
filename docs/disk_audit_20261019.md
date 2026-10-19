# hw3090 磁盘审计报告（2026-10-19，READ-ONLY）

审计范围：root NVMe（`/dev/mapper/vgubuntu-root`，挂载 `/`）与 HDD（`/mnt/data`）。本次审计没有删除、移动、压缩、上传任何东西，没有运行 rclone，没有终止任何进程。所有大小来自 `du -x`（十进制/二进制近似，约 +-5%）。docker 的数字来自 `docker system df -v`（read-only）。

## 1. 当前 df

| 文件系统 | 总量 | 已用 | 可用 | 使用率 |
|---|---|---|---|---|
| `/`（NVMe，LVM） | 914G | 740G | 128G | 86% |
| `/mnt/data`（HDD，TOSHIBA 7.3T） | 7.3T | 1.6T | 5.4T | 23% |
| `/tmp`、`/home` | 与 `/` 同一文件系统 | | | |
| `/dev/shm` | 32G tmpfs | 48K | | 1% |

结论：压力只在 root NVMe；HDD 余量充足（5.4T），HDD 上的清理不是紧迫项。

### 1.1 root NVMe 占用去向（740G）

| 去向 | 大小 | 备注 |
|---|---|---|
| `/home/xinyuan` | 363G | 见第 3、4 节 |
| docker images（`/var/lib/docker`，root-only，不可读） | 约 125-140G（估算） | 11 个不同的 25GB 级镜像约 120G，共享 base 层 15.8G，其余小镜像、`ol7-xfce-vnc` 6.3G |
| 其他用户 home（`/home/hw3090`、`/home/ly`、`/home/jiangjizhong`，权限 750） | 约 200G（推算，无法读取） | `du` 报 Permission denied；需要 sudo 才能确认，不在本审计范围，不能动 |
| 系统（`/usr` 22G、`/var` 8.7G、`/opt` 4.3G、`/tmp` 3.1G） | 约 38G | `/var/log/journal` 3.9G（journalctl 报 1.0G）；`/usr/local/cuda-12.2` + `cuda-12.4` 9.4G |

说明：740G - 363G - 38G - docker 约 130G，约 200G 去向不明，推测为其他三个 Linux 用户的 home。这部分需要 MZ 决定是否让相应用户自行清理，或用 sudo 查看。

## 2. 活跃进程与不能碰的路径

- 运行中的 `python`：`/home/xinyuan/si10mad_fp/ws/experiments/mea_multiseed_robustness_20261018_v1/py3916/bin/python ... mea_all_methods_devretuned_20261019_v1`（2 个进程，各占 14-18GB RSS，CPU 100%）。另有 2 个 `pi` worker、`agy`、`claude --resume`、`zcode-cli`；`nvidia-smi` 当前没有 compute app。
- 严格不碰：`SSM/results/pi_logs`（目前只有 `p0solve_20261019.log`、`streamf7_20261019.log`；`results/p0solve_20261019` 和 `results/stream_f7_20261019` 目录目前尚不存在，输出可能写到别处，后续会创建）。
- docker 有 2 个运行中的容器（`dengyi_eda_ol7`、`xinyuan_eda_ol7`，`ol7-xfce-vnc`，up 3 days），其中 `dengyi_eda_ol7` 看起来是别人的 EDA 环境，不要动。

## 3. 分类候选清单：root NVMe

大小单位 GB（G）。"证据" 列写明是否 git 追踪、是否被当前代码/文档引用、mtime、是否属于活跃项目。

### (A) 可直接删除（可再生的缓存或失败产物）

| 路径 | 大小 | mtime | 是什么 | 证据 | 建议 |
|---|---|---|---|---|---|
| `~/.cache/pip` | 3.3G | 今天 | pip http cache | 纯缓存，`pip` 会重建 | A：`pip cache purge` |
| `~/.triton/cache` | 1.3G | 今天 | Triton 编译缓存 | 1596 个条目，重建只需重新编译 | A（正在运行的 job 会重新编译，建议等当前 job 结束后） |
| `~/.npm/_cacache` | 0.62G | | npm cache | 纯缓存 | A：`npm cache clean --force` |
| `~/miniconda3/pkgs` | 5.8G（与 envs 不共享的部分；`du` 单独看是 13G，其余是与 envs 的 hardlink） | | conda 包缓存（含 `cudatoolkit-11.2.2` 1.6G、`cudnn-8.1.0.77` 1.4G、多个旧 python） | 只在 hardlink 以外的部分才会释放；`conda clean --all` 对在用 env 是安全的 | A：`conda clean --all -y`，预计释放约 5.8G |
| `/tmp/triton_*`（13 个目录：`pc3fix_gpu0_laneA..D`、`drift_1..3`、`hybrid`、`hybrid2`、`hyb2`、`semgfix`、`bl3`、`baselines`） | 约 2.4G | 10-16 至 10-17 | 已结束 lane/job 的 Triton 缓存 | 这些 job 都已结束，`docs` 里只有命令行历史引用 | A |
| `/tmp/pytest-of-xinyuan` | 225M | 10-17 | pytest 临时目录 | 可再生 | A |
| `/tmp/bl_review`、`bl_review2`、`bl_dfix`、`expv2_review`、`exp_smoke*`、`semg_review`、`m2`、`m2_review`、`m2_analysis_before`、`hyb_review`、`hyb2`、`r1`、`rd`、`arch_an`、`exp_dry`、`exp_review`、`semgfix_work`、`build`、`colab_test`、`nomad_data_backup_543614`、`fia_test_*`（约 40 个空目录）、`__pycache__`（7.5M）、`c*.py`、`edit*.py`、`queue_*.sh` 等小文件 | 合计约 0.4G（`bl_review` 86M，`m2` 39M，`nomad_data_backup_543614` 9M） | 10-16 至 10-18 | 老的 review/smoke 临时目录和临时脚本 | review 已完成；`/tmp/m2` 的 manifest 已落入 `results/analysis/m2_calib_20261016`（可 diff 确认） | A（删除前对 `/tmp/m2`、`/tmp/nomad_data_backup_543614` 做一次 diff/确认） |
| `/tmp/pi-*.log`、`/tmp/pi-subagents-uid-1002`、`/tmp/claude-1002`、`/tmp/node-compile-cache`、`/tmp/jiti`、`systemd-private-*` | 约 30M | 今天 | 正在运行的 pi/claude 进程使用 | 不要碰 | C |
| `~/Work_host/SSM/.tools/triton_peft_gpu1` | 150M | 10-03 | Triton 缓存，位于 SSM 内（git-ignored） | `.gitignore` 已忽略 `.tools/` | A |
| `~/Work_host/**/__pycache__`、`.pytest_cache`、`numba_cache*`（共约 1200 个 `__pycache__` 目录，含 iscas、`np_*`、`si10mad_fp`） | 约 0.23G | | Python/numba 缓存 | 可再生 | A |
| `~/Work_host/iscas/autonomous_canonical_noise300_20261014_v1/attempts/`（含 `white_independent_partial_attempt1.f32` 与 `full_generation_attempt1_partial_output`） | 32G | 10-14 | 生成 300 s 噪声失败的 attempt 1/2 残留（`rss_exceeded`、`validation_attempt{1,2}_error.txt`） | 成功的 `output/`（23G，read-only 且有 `IMMUTABLE_MANIFEST.json`）已产出；`attempts/*.txt` 仅是几 KB 的失败记录 | A，但建议保留 `attempts/*.txt` 小文件，只删 `.f32` 和 `full_generation_attempt1_partial_output/`。该目录不在 git，MZ 请确认 |
| `~/.vscode-server/cli/servers/Stable-*`（9 个版本，共 5.7G；`lru.json` 记录最近用的 5 个） | 约 4.4G 可释放（保留 `lru.json` 前 2 个） | | 旧版 VS Code Server | 当前没有 `vscode-server` 进程；它们会在需要时重新下载 | A（只保留最新 2 个） |
| `~/.cursor-server/bin/linux-x64/*`（4 个版本，共约 1.6G） | 约 1.2G | | 旧版 Cursor server | 同上 | A（保留 1 个） |
| `~/.local/share/cursor-agent/versions/*`（3 个版本，1.6G） | 约 1.05G | | 旧版 cursor-agent | 保留最新 `2026.09.15-d2fe57e` | A |
| `~/.nvm/versions/node/v24.16.0` | 0.42G | | 老 node（默认是 v26.1.0） | 默认 alias 为 v26 | A（可选） |
| `~/.vscode-server/data/CachedExtensionVSIXs` | 235M | | 扩展安装包缓存 | 可再生 | A |
| journal（`/var/log/journal`） | 1.0G（journalctl 值） | | systemd journal | root 权限，需要 `sudo journalctl --vacuum-size=200M` | A，需要 sudo |
| `~/Work_host/APST-open-source-20260915.tar.gz` | 68M | 09-15 | APST 开源包 tarball | 内容与 `Work_host/APST`（git，remote `HsinyuanZhang/APST`）和 `APST-public` 重复；GDrive 上是否存在副本，本次审计不能运行 rclone 检查 | A/B：确认 GDrive 有副本（`rclone lsf`）后删；否则先上传 |
| `~/cc-switch-cli-linux-x64-musl.tar.gz` | 5.6M | 05-19 | 安装包 | 可再下载 | A（可选） |
| `~/Work_host/Eqm_removal_tmp`、`~/Work_host/sua_exploration`（空壳 12K） | <1M | | 空目录 | | A（可选） |

**A 小计（root）**：约 55G，其中失败 attempts 32G、工具缓存 11G、旧 IDE 版本 7G、/tmp 3G、其余 2G。

### (B) 打包后归档到 GDrive（或搬 HDD），确认后再删本地

注意：从 rclone 日志看，上传速度只有约 1-2 MiB/s（`results.tar.zst` 13.5G 花了数小时），而且这些大文件主要是生成的随机数据（`.h5`、`.f32`、`.npz`），压缩比很差。因此对 >10G 的可再生数据，更实际的选择是：搬到 HDD（`/mnt/data/work_host/` 下建子目录，原位置留 symlink，沿用 2026-10-02 的迁移约定），而不是上传 GDrive；GDrive 只放 compact 结果（json/csv/log/小 checkpoint）。

| 路径 | 大小 | mtime | 是什么 | 证据（是否仍被使用） | 建议 |
|---|---|---|---|---|---|
| `Work_host/iscas/autonomous_canonical_noise300_20261014_v1/output` + `input` | 23G | 10-14 | 300 s 10uV rms 噪声文件（`canonical_independent_300s_10uVrms.f32` 24.6GB，read-only） | 目前 `si10mad_fp/ws/experiments` 中没有文本引用；有 manifest，是 seed 生成的 | B（搬 HDD）或 A（若 manifest + 生成脚本足够重现） |
| `Work_host/iscas/autonomous_dense_mea_20261014_v1` | 30G（`output` 24G，`input` 5.6G） | 10-14 | dense MEA 1024ch/64u/300s `.h5` | 同上，无引用 | B（搬 HDD） |
| `Work_host/iscas/autonomous_dense_mea_continuous_20261014_v2` | 24G | 10-14 | continuous 版本 `.h5` | 同上 | B（搬 HDD） |
| `Work_host/iscas/autonomous_20261013T234634_np` | 9.7G（`mechanisms_np_spatial_drive_20261014` 7.2G，`round2_20261014` 1.7G） | 10-14 | NP spatial drive 机制/round2 | 其副本 `si10mad_fp/ws/experiments/autonomous_20261013T234634`（327M）为后续使用的版本 | B（搬 HDD，保留小 json） |
| `Work_host/iscas/spatial_mechanism_ablation_20261016_v1` | 7.2G（`np/full_meddrift_v2` 6.8G） | 10-16 | spatial 机制 ablation | 无后续引用 | B（搬 HDD） |
| `Work_host/iscas/` 其余约 40 个 round 目录（`lc6_*`、`np_*_round*`、`neuropixels_*`、`np_*_opt_20261011` 等，各 40-580M） | 约 3.5G | 10-09 至 10-16 | 已完成的 round | 无活跃进程使用；`dataset_usage.md` 与 `results/` 为说明文件 | B（pack `tar -I zstd` 上传 GDrive:DAC27，验证后删；本身体量小，可上传） |
| `Work_host/iscas/np_mea_aligned_20261013`（20G） | 20G | 10-13 | NP/MEA aligned 实验 | **被 `si10mad_fp` 的 CONTRACT/INPUT json 引用 35 次，当前 job 的上游输入** | C（保留；等 si10mad_fp 项目结束后再评估） |
| `Work_host/iscas/neuropixels_drift_comparison_20261009`（301M）、`neo_neighbor_20261016_v1`（318M） | 0.6G | | | 被 `si10mad_fp` 引用（37 次/2 次） | C |
| `~/np_*_2026101*`（`np_soft_neighbor_..._v1` 1.2G、`np_fp_mechanism_..._v1` 187M、`np_unified_DEV_..._v2` 37M、`np_physical_nms_..._v1` 8.9M、`np_q10_same_span_DEV_..._v1` 3M、`np_unified_DEV_..._v1` 2.7M）、`~/spatial_algorithm_to_rtl_20261015T2233_v1`（281M）、`~/np_fixed_spatial_v1` | 约 1.7G | 10-15 至 10-16 | 一批 NP 机制 DEV 目录，含 `numba_cache`、`CONTROLLER_EXIT.json` | 已结束（`CONTROLLER_EXIT.json`），其内容是 iscas 项目的前序 round | B（pack 后上传，小） |
| `~/rt_seed_robustness_annex_v2_1_control_20260810_v3` | 156K | 08-11 | receipts | 很小 | C |
| `Work_host/SPINT`（git，remote `HsinyuanZhang/iBCI`，最后提交 2026-09-12）本地非 symlink 部分：`sua_exploration/m1_compact_replication/results` 1.3G、`sua_exploration/cache` 795M、`sua_exploration/evalai_m1_threeway` 363M、`SPINT-main` 1.0G（`data` 已是 symlink）、`btransform_unified_v1` 155M、`btransform_unified_v2` 291M、`tfpd_exploration` 338M、`cebra_exploration` 148M、`software-to-hardware` 112M、`bci_paper_overleaf` 284M、`apst_final_week_20260916` 163M | 约 6G | 07-31 至 10-03 | 旧项目（SPINT/APST，9 月）；`git status` 有 881 项改动（大量被删的 tracked 文件，如 `btransform_unified_v2/dandi688_bench_v2/*`，因为目录已替换为 symlink 到 HDD） | B：对 `m1_compact_replication/results`（三个 `*_pull_20260810_*` 目录各约 440M，疑似重复拉取）、`cache`、`evalai_*/artifacts` 做 `tar -I zstd` 后上传，验证后删除；代码和 git 保持原位 |
| **`Work_host/SPINT/sua_exploration/data/dandi_000688`（sub-C 9.1G、sub-M 2.2G、sub-J 84M）+ `data/000128`（662M）、`000129`（49M）** | 12G | 07-22 至 08-05 | DANDI 原始 NWB 数据 | **SSM 当前代码直接读取**：`ssm_decode/dandi688_data.py`、`scripts/run_expansion.py`、`ssm_decode/expansion_protocol.py` 等引用 `SPINT/sua_exploration/data/dandi_000688/sub-C`。按 AGENTS 规则，数据集应在 HDD | C，但建议 MZ 决定是否搬到 HDD 并留 symlink（HDD 上已有 `work_host/datasets/DANDI_000688_missing` 1G，但不含 sub-C/M/J 全量）。**不要在有 job 运行时搬动** |
| `~/SNN_SpikeSorting/.worktrees/exp-{reservoir,soft-spike-bottleneck,morph-teacher}-ae` | 5.8G（2.4G/1.8G/1.8G） | 最后提交 06-01 | 三个 stale git worktree（含 `new_datasets` 703M 副本、`SpikingJelly_SS` 999M） | 三个 worktree 各有 3-16 个未提交改动；另有两个 prunable worktree（`~/SNN_SpikeSorting_sparse_expansion`、`~/SNN_SpikeSorting_tiny_encoder`，目录已不存在，`git worktree prune` 即可） | B：先对未提交改动打 patch，pack worktree（排除 symlink 数据）上传，验证后删 |
| `~/SNN_SpikeSorting` 本体（`.git` 1.7G、`SpikingJelly_SS` 1.1G、`Wave_clus` 325M、`Quiroga` 184M、`Spatial` 173M 等） | 约 3.7G | 05 至 10-07 | 早期 spike sorting 项目；数据已通过 symlink 指向 HDD | `DAC2027/` 子目录（5.4M）可能相关 | C（保留）；`.git` 1.7G 可 `git gc` |
| `~/.codex/sessions`（4.6G）+ `thread_history_1.sqlite`（618M）+ `~/.codex/packages/app-server-daemon`（3.0G） | 8.2G | 10-16 | Codex 会话历史与 app-server 包 | 历史只用于回看；`packages` 可重新下载 | B：sessions/history 打包上传 GDrive；`packages` 视为 A（需先确认没有 codex 进程在用，目前未看到） |
| `~/.cursor/chats`（4.3G，主要是 `3c58...` 4.1G）、`~/.cursor-server/data/snapshots`（2.6G） | 6.9G | | Cursor 聊天/快照 | 无 | B/A，MZ 决定是否还需要 Cursor 历史 |
| `~/.rustup`（1.4G）+ `~/.cargo`（531M）+ `~/DeepSeek-TUI`（1.3G，rust 源码 + `target`） | 3.2G | 05-15 | Rust 工具链与 DeepSeek-TUI 源码 | 非 DAC2027 | A/C，MZ 决定 |
| `~/.zcode`（1.3G）、`~/.deepseek`（1.2G）、`~/.grok`（659M）、`~/.qoder`（586M） | 3.7G | | 其他 AI 工具的下载缓存/二进制 | 其中 `zcode-cli` 正在运行 | C（不动，除非 MZ 明确不再用） |
| `~/DATE2027`（119M，`vqvae-adc-handoff` 等） | 119M | 09-18 | 另一论文项目 | | C |
| `~/Work_host/APST`（457M，git，remote `HsinyuanZhang/APST`）、`APST-public`（71M，含 `checkpoints`）、`ibci_c3_film`（186M，git）、`nomad`（28M，git；`SSM/scripts/run_baselines_nomad_*` 引用）、`context_general_bci`（11M，git） | 约 0.75G | | 小型代码库/checkpoint | 都有 git；`APST/src` 被 `SSM` 引用 2 次，`Work_host/nomad` 被引用 1 次 | C |
| `~/miniconda3/envs/snn_cuda`（5.9G，atime 10-17，mtime 05-14） | 5.9G | | SNN 项目 conda env | 10-17 仍被访问；`spint` env（12G）是 SSM 脚本使用的 python（`miniconda3/envs/spint/bin/python` 在 `scripts` 中被引用 29 次） | `spint` 为 C；`snn_cuda` 需要 MZ 决定 |
| `~/.local/lib/python3.10/site-packages`（4.9G，含 `nvidia` 2.7G、`torch` 1.2G） | 4.9G | | 用户级 pip 安装（系统 python3.10） | 不确定是否被某些脚本依赖 | C，需要 MZ 决定 |
| docker images（见第 5 节） | 约 125G | | FALCON 提交镜像 | 见第 5 节 | 需要 MZ 决定 |
| `~/si10mad_fp`（108G） | 108G | 10-15 至 今天 | ISCAS spike-detection 项目的运行区；`ws/experiments/mea_multiseed_robustness_20261018_v1` 90G（`datasets/D1-D4` 各 22G + `py3916` 环境），`unified_neighbor_priority_20261018_v1` 7.0G，`hard_neighbor_dev40_20261018_v1` 4.3G，`delta_ema_aso_sw_test_20261018_v1` 3.4G | **有 2 个活跃 python 进程在使用，mtime 今天** | C（不要动）。待项目结束后，`D1-D4`（87G）的生成数据可考虑搬 HDD |

**B 小计（root，不含 docker、不含 C 项）**：

- iscas 大型生成数据（搬 HDD）：约 94G（23+30+24+9.7+7.2）
- iscas 小 round + 家目录 np_*：约 5.2G（可上传 GDrive）
- SPINT 本地非 symlink 数据：约 6G
- SNN_SpikeSorting worktrees：5.8G
- agent 历史（codex sessions/history、cursor chats/snapshots）：约 15G（其中 packages 3.0G 视为 A）
- 合计约 126G（若加 docker 约 125G，则约 250G）

### (C) 保留

| 路径 | 大小 | 理由 |
|---|---|---|
| `Work_host/SSM/results/*`（除下节分析的候选） | 7.0G 中约 4G | 当前 DAC2027 分析使用（见第 4 节） |
| `Work_host/SSM/results/pi_logs`、以及 `p0solve_20261019`、`stream_f7_20261019` 输出 | 小 | 活跃 job，禁止触碰 |
| `Work_host/SSM/.tools/{mamba_deps,mamba_official,memba_upstream,ssm_peft_upstream,iverilog}` | 约 680M | 当前 SSM 脚本依赖（`mamba_deps` 被引用 14 次） |
| `Work_host/SSM/.git` 22M | | |
| `~/si10mad_fp`（108G） | | 活跃项目 |
| `Work_host/iscas/np_mea_aligned_20261013`、`neuropixels_drift_comparison_20261009`、`neo_neighbor_20261016_v1` | 20.6G | 被 si10mad_fp 引用 |
| `~/miniconda3/envs/spint` | 12G | SSM 的 python |

## 4. `Work_host/SSM/results/*` 逐项评估（总计 7.0G）

注：`.gitignore` 规则 `results/*`，只追踪 `results/analysis/`（580 个文件）、`debug_audit`（3）、`peft_round1`（35）。除此之外 `results/*` **没有被 git 追踪**，也没有备份，唯一的保护是 `DAC27_backup/SSM_old_20261015/results.tar.zst`（14.5G，已在 GDrive 验证；我用 `zstd -dc | tar -t` 流式读取了前几十个顶层条目，确认包含 `forward_eval_m2_wide_n16*`、`stage2_688_20261005`、`stage2_intervals_20261006`、`e1b_m2_20261007`、`labeled_calibration_20261007`、`data_cache`、`peft_round1`、`debug_audit`、`analysis`；这是 10-15 19:17 的快照，所以 10-15 之后新建的 db6/hyser/baselines/population_calib/hybrid/semg/m2_calib/drift_curve/expansion 目录没有备份）。

| 路径 | 大小 | mtime | 内容 | 被代码/文档引用？ | 建议 |
|---|---|---|---|---|---|
| `data_cache/`（db6 1.0G、hyser_1dof 747M、m1 213M、m2 44M） | 2.0G | 10-04 至 10-15 | 窗口化 npz 缓存 | 9 个脚本/文档引用，当前脚本使用 | C（可由 HDD 数据再生，但耗时；保留） |
| `baselines_20261016/apst2014/runs`（`sua_dual_site_F0_s42` 458M、`sua_concat_pretrain_s42` 457M） | 925M | 10-16 | APST 2014 retrain checkpoint | `run_baselines_apst.py`、`analyze_baselines.py` 引用 | C，当前 baseline（paper 用）；paper 提交后可 B |
| `baselines_20261016/nomad/models`（`days`、`y2014`、`y2014_pool`、`days_pool` 各约 165M）+ `preds` | 757M | 10-17 | NoMAD 模型 | `analyze_baselines.py` 读取 `nomad/controls_*.json` 和 preds | C；`models` 661M 为再生可行（nomad 训练），论文定稿后 B |
| `baselines_20261016/v1_pre_review` | 28M | 10-17 | review 前的 v1 结果（含 `nomad_preds_legacy`） | `analyze_baselines.py` 有 `nomad_legacy_table()` 引用 | C（小，且被引用） |
| `baselines_20261016/eval_pre_dfix`、`eval_pre_fix`、`eval_pre_native`、`eval_old`、`eval_aux` | 约 3M | 10-16/17 | 被取代的 eval 版本 | 旧版，已被 `eval/` 取代 | A/B：小到可忽略，建议随 `v1_pre_review` 一起 pack 归档 |
| `baselines_20261016/logs` | 12M | | 日志 | | C |
| `baselines_20261016_selsens` | 2.9M | | selection sensitivity | | C |
| `forward_eval_m2_wide_n16_20261005`、`_s1_20261006`、`_s2_20261006` | 886M+885M+853M = 2.6G（`targets` 约 760M 各，`pools` 95-126M 各，含 `best.pt`/`session_bank_*.pt`） | 10-15 20:31（从旧快照还原） | Stage0 forward eval 的 target/pool checkpoint | 文档 `stage0_forward_eval_20261005.md`、`labeled_calibration_results_20261007.md` 提及；脚本引用 1-3 次；**同一内容在 `SSM_old_20261015/results.tar.zst`（GDrive 已验证）中** | B：可直接删除本地 `targets/` 与 `pools/`（保留 `summary.json`、`plan.json`），因为已有 GDrive 验证过的备份。需要 MZ 批准 |
| `stage2_688_20261005`、`stage2_intervals_20261006`、`e1b_m2_20261007`、`labeled_calibration_20261007` | 18M+21M+10M+5M | 10-15 | 已完成 round | 被 `analyze_track_a.py` 等引用；也在旧 tarball | C（小，保留） |
| `db6_20261015`（v1，113M） vs `db6_v2_20261016`（57M） | 170M | 10-15/16 | NinaPro DB6 的 v1 与 v2 结果 | `compare_db6_v1_v2.py` 同时引用，两者都被文档引用 | C（小，且 v1/v2 对比还要用） |
| `population_calib_20261015` (p1，24M)、`population_calib_p2_20261015`（127M，`fix` 109M）、`population_calib_p3_20261016`（129M，`select` 81M、`fix` 41M） | 280M | 10-15 至 10-16 | 三轮 population calibration | 脚本 `analyze_pc2.py`、`analyze_pc3.py`、`pc3_fix_cpu.py` 等 14-20 次引用；p3 是当前版本 | C（小）。若要瘦身：p2 `fix`（109M）已被 p3 取代，可 B |
| `hybrid_identity_20261016`（11M，v1）、`hybrid_identity_v2_20261017`（35M） | 46M | 10-16/17 | hybrid v1 与 v2 | `analyze_hybrid_identity{,_v2}.py` 都引用 | C |
| `hyser_20261015`（72M）、`semg_calib_20261016`（55M）、`m2_calib_20261016`（31M）、`drift_curve_20261016`（2.2M）、`expansion_round1`（43M，含 2 个 tgz）、`expansion_20261016`（空）、`peft_round1`（616K，git 追踪）、`debug_audit`（git 追踪）、`analysis`（27M，git 追踪） | 约 230M | | 当前分析/论文用 | 均被引用 | C |
| `pi_logs` | 12K | 今天 | 活跃 worker 日志 | | C，不碰 |

**SSM/results 可回收**：`forward_eval_*` 约 2.6G（B，已有 GDrive 备份）；其余 <0.2G。SSM 总体约 7.9G（`results` 7.0G、`.tools` 833M），其中大部分需要保留。results 中**不是**root NVMe 压力的来源。

另外要提醒：10-15 之后新增的 results（db6、hyser、baselines、population_calib_p*、hybrid、semg、m2_calib 等，合计约 3.5G）目前只在本地，没有 GDrive 备份、没有 git 追踪。按"GitHub 同步有意义的 csv/json"的分工，建议后续把这些 results 中的紧凑 json/csv 同步到 git，把 checkpoint（`apst2014/runs`、`nomad/models`，共约 1.6G）pack 后上传 GDrive。

## 5. Docker（`docker system df`，read-only）

- Images：20 个 repo:tag 条目（`docker system df` 报告 total 36.6GB，但这是虚高：reclaimable 显示为负数，说明 containerd 的 overlayfs snapshot 统计口径不一致）。按 `docker images` 实际按 image ID 去重后：
  - 11 个不同的 25.2-25.3GB 镜像（FALCON 提交镜像，创建于 09-18 至 09-22，已推送到 ECR `...falcon-2319-participant-team-42279/42308`，本地同时带 `m1-*`/`h1-*`/`spint-m2*` 别名标签）。共享层约 15.8G，每个独享约 9.5G，因此这 11 个合计约 120G。别名标签（`m1-x16-...`、`ECR-...`）指向同一 image ID，不额外占空间。
  - 6 个约 2.0-2.3GB 的镜像（`ssm-falcon-m1/m2`、`ssm-falcon-cpu-runtime`、`apst-m1/m2/h1:public-rebuild`，10-01 至 10-03）共享 1.6G base，合计约 5G。
  - `python:3.10-slim-bookworm` 195M；`ol7-xfce-vnc:latest` 6.3G（被 2 个运行中的容器使用，是 EDA 环境，不要动）。
  - 大量 `<none>` dangling 条目只是 `docker images -a` 列出的中间层，不是独立占用。
- Containers：4 个（2 个 EDA 运行中，2 个 `ssm-m{1,2}-sdk-preflight-20261003` 已退出，约 20K）。
- Volumes 0，Build cache 0。
- Docker root 在 `/var/lib/docker`，需要 root 才能读，无法用 `du` 精确核对。

建议：
- 11 个 25GB FALCON 提交镜像已推送到 ECR；`Work_host/APST`（commit "Publish final FALCON checkpoints and reproducible Docker builds"）和 `APST-public/docker` 已有可复现的 build，因此本地副本可视为可再生（约 120G）。这是单项最大的可回收项。但挑战赛结果与"最终提交使用的镜像"是否需要保留本地副本、以及 ECR 是否仍保留这些 tag，需要 MZ 确认。最保守的方案是保留 1-2 个最终提交镜像（`m1-x16-s8-lr250-d006-m4-hidden:v1` 等 hidden 提交），`docker rmi` 其余；若要归档，`docker save | zstd` 每个约 10-13G，按 1-2 MiB/s 的上传速度不现实，不建议上传 GDrive。
- 2.0-2.3GB 的 `ssm-falcon-*`、`apst-*:public-rebuild` 属于 DAC2027 前的 SSM 提交与重新构建，保留（C）。

## 6. HDD `/mnt/data`（1.6T，5.4T 可用）

| 路径 | 大小 | mtime | 内容 | 证据 | 建议 |
|---|---|---|---|---|---|
| `datasets/hd-semg`（143G）、`datasets/ninapro_db6/extracted`（21G） | 164G | 10-15 | Hyser、NinaPro DB6 解压数据 | SSM 脚本引用 `/mnt/data/datasets/hd-semg`、`ninapro_db6/extracted`；hd-semg 已 sha256 校验 | C（数据集保留） |
| `datasets/ninapro_db6/DB6_s*_{a,b}.zip`（20 个，共约 21G） | 21G | 2017（原始）/ 10-15 下载 | NinaPro 原始 zip | 已有 `extracted/`（20 个目录），zip 可重新下载 | A（可选）：MZ 决定。`extracted` 完整性我没有校验（`unzip -n` 无 checksum），删除前应先对比 zip 内文件列表 |
| `work_host/datasets`（75G：Churchland_2012 50G、ODoherty_2017 23G、FALCON_H2 1.2G、DANDI_000688_missing 1.0G、NLB_MC_Maze*） | 75G | 10-06 | 公共数据集 | `MANIFEST.md` 存在 | C |
| `work_host/SNN_SpikeSorting`（92G：Hybridjanelia 60G、synthetic_spike_dataset 26G、new_datasets 3.2G、MEArec 1.8G、neuropixel 1.4G） | 92G | 10-02 | spike sorting 数据集（`~/SNN_SpikeSorting/*` 的 symlink 目标） | `si10mad_fp`/iscas 使用 | C |
| `backup_datasets/SNN_SpikeSorting/{Hybridjanelia,synthetic_spike_dataset,new_datasets,MEArec,neuropixel}`（共约 92G） | 92G | 07-26 | 2026-07-26 的备份副本 | **与 `work_host/SNN_SpikeSorting` 同盘、同内容重复**（README 记录：备份后原位置才迁移到 `work_host`；2026-10-02 迁移时用 rsync 校验零差异）。两份都在同一块 HDD，没有真正的灾备价值 | 需要 MZ 决定：保留以防 `work_host` 副本被误删，或删除重复的一份。若删除，应先在 GDrive 看有没有备份（本次无法确认） |
| `backup_datasets/SNN_SpikeSorting/figshare_25298815_kilosort4_sims` | 320G | | Kilosort4 论文仿真数据（342G） | `iscas/dataset_usage.md` 列为主要数据；**只有这一份** | C（唯一副本，保留） |
| `backup_datasets/SNN_SpikeSorting/zenodo_1205233_ground_truth`（34G）、`zenodo_1205233_work`（7.8G）、`.tmp_*` | 42G | | Zenodo 1205233 视网膜 patch 数据 | 唯一副本 | C |
| `backup_datasets/EDA`（92G，`cadence` 92G、`dengyi`、`synopsys`、`tech`、`proj`） | 92G | 05-15 | 原 `/home/xinyuan/EDA` 的备份（含 13355 个 symlink，其中 777 个原本断裂） | 与 DAC2027 无关，属于 EDA 工具/环境，且与其他用户（`dengyi`）有关 | 需要 MZ 决定；不动 |
| `SPINT_cold_archive`（424G：`2026-08-13` 134G、`2026-09-11` 87G、`2026-09-12` 57G、`2026-09-16` 118G、`2026-09-16-pass2` 29G） | 424G | 08-13 至 09-16 | 旧 SPINT/btransform/sua/tfpd 的 checkpoint 与 results 冷归档（`MANIFEST.txt` 在 09-11、09-12 中） | 旧项目；GDrive 是否有副本不详；多个 snapshot 可能互相有重叠（例如 `btransform_unified_v2` 在 4 个批次中各出现一次：118G、38G、19G、11G，内容路径不同） | 保留；若 HDD 紧张时，这是最大的清理候选，但需要先在 GDrive 做压缩备份（424G 以 1-2 MiB/s 上传需要约 2-4 天，不现实）。建议 MZ 决定该项目是否已经完成 |
| `work_host/SPINT`（242G：`apst_final_week_20260916` 157G、`btransform_unified_v2` 38G、`tfpd_exploration` 31G、`SPINT-main/data` 17G） | 242G | 10-02 | `Work_host/SPINT` 中 symlink 的 HDD 目标 | symlink 目标，**不要删**（`rm` 符号链接不删数据，反之亦然）；`SSM` 脚本引用了 `work_host/SPINT/SPINT-main`（6 次）、`apst_final_week_20260916`（2 次） | C |
| `work_host/envs/nomad_tf26`（4.8G） | 4.8G | 10-16 建立 | NoMAD 的 TF 2.6 env | SSM `run_nomad_jobs.sh`、`run_baselines_nomad_score.py` 使用 | C |
| `work_host/envs/nomad_tf`（5.1G，TF 2.15.1） | 5.1G | 10-16 | 第一版 NoMAD env | 脚本里没有对 `envs/nomad_tf/` 的引用，只用 `nomad_tf26`；atime 今天只是 `du`/`ls` 访问 | A/B：MZ 确认不再需要后可删（可 pack 成 tar.zst 约 2G 上传） |
| `work_host/envs/*.log`、`migration_log_20261002.txt`、`README.md` | <1M | | | | C |
| `DAC27_backup/SSM_old_20261015`（14G：`results.tar.zst` 14.5G、`tools.tar.zst` 235M、`SSM_full.bundle` 2.4M） | 14G | 10-15 | 旧 SSM 的 tarball | AGENTS 记录已在 GDrive 验证；rclone 日志结尾显示 `Copied (new)`，100%；**MZ 尚未批准删除**。这个 tarball 也是上面 SSM/results 中 `forward_eval_*` 的唯一备份来源之一 | 待 MZ 批准；在批准之前，C。批准后 A（但建议先保留到 `forward_eval_*` 本地副本也不再需要为止） |
| `DAC27_backup/colab_stage/SHA256SUMS_round2`、`rclone_*.log` | <1M | | | | C |

**HDD 可回收小计**：A 约 26G（ninapro zip 21G + `nomad_tf` 5.1G，需要 MZ 同意），待批准 14G（`SSM_old_20261015`），需要 MZ 决定约 92G（backup_datasets 重复副本）和 92G（EDA）。HDD 并不紧张，建议当前不做大动作。

## 7. 汇总：可回收量

### root NVMe（现在 128G 可用，86%）

| 类别 | 估计可回收 | 说明 |
|---|---|---|
| (A) 直接可删 | 约 55G | 失败 attempts 32G + 工具缓存 11G + 旧 IDE/Cursor/node 版本 7G + /tmp 3G + 其他 2G（journal 1G 需 sudo） |
| (B) 打包归档后删 / 搬 HDD | 约 126G | iscas 大型生成数据 94G（搬 HDD）、agent 历史 15G、SPINT 本地 6G、worktrees 5.8G、iscas 小 round 与家目录 np_* 5.2G |
| docker 镜像（需 MZ 决定） | 约 120G（11 个 25GB 镜像） | 已推送 ECR，APST 有可复现 build |
| (C) 保留 | 其余 | 其中 `si10mad_fp` 108G 为活跃项目 |
| 无法检查 | 约 200G | 其他用户 home（需要 sudo） |

如果执行全部 A + B（不含 docker）：约 180G，使用率 86% 降到约 67%；再加 docker 可降到约 54%。

### HDD（5.4T 可用，23%）

| 类别 | 估计可回收 |
|---|---|
| (A) 可删 | 约 26G（ninapro zip 21G + `nomad_tf` 5.1G，MZ 同意后） |
| 待 MZ 批准 | 14G（`DAC27_backup/SSM_old_20261015`） |
| 重复，需 MZ 决定 | 约 92G（`backup_datasets/SNN_SpikeSorting` 与 `work_host/SNN_SpikeSorting` 重复） |
| 不相关，需 MZ 决定 | 92G（`backup_datasets/EDA`）；424G（`SPINT_cold_archive`，旧项目） |
| (C) 保留 | 数据集 `datasets`（184G）、`work_host/datasets`（75G）、figshare（320G）、zenodo（42G）等 |

## 8. 需要 MZ 决策的事项

1. **docker 25GB 镜像（约 120G）**：是否可以 `docker rmi` 11 个已推送到 ECR 的 FALCON 提交镜像？是否保留 hidden 最终提交的 1-2 个？这是最大的单项。
2. **`iscas/autonomous_canonical_noise300_20261014_v1/attempts`（32G）**：失败 attempt 残留，可删；是否保留 `.txt` 失败记录。
3. **iscas 大型生成数据（约 94G）**：搬 HDD（建议）还是直接视为可再生；同时 `si10mad_fp` 项目结束后，`mea_multiseed_robustness_20261018_v1/datasets/D1-D4`（87G）怎么处理。
4. **`SPINT/sua_exploration/data/dandi_000688`（12G）**：SSM 当前直接读取；是否搬到 `/mnt/data/work_host/datasets` 并留 symlink（需要在没有 job 时进行，并需要改动 `data_cache` 里的路径吗？当前代码用的是原路径，symlink 后不需要改）。
5. **`DAC27_backup/SSM_old_20261015`（14G）删除**：是否批准；注意它也是 `forward_eval_*` 的唯一备份。
6. **`forward_eval_m2_wide_n16*`（2.6G）**：是否同意删除本地 `targets/` 与 `pools/`（备份在 GDrive 验证的 tarball）。
7. **HDD 重复副本**：`backup_datasets/SNN_SpikeSorting`（92G，同盘重复）是否保留。
8. **`miniconda3/envs/snn_cuda`（5.9G）、`~/.local/lib/python3.10/site-packages`（4.9G）、`.rustup`/`DeepSeek-TUI`（3.2G）、agent 历史（`.codex/sessions`、`.cursor/chats`）**：是否仍需要。
9. **`work_host/envs/nomad_tf`（5.1G，TF 2.15.1）**：脚本只用 `nomad_tf26`，是否可删。
10. **其他三位用户的 home 占用约 200G**：需要 sudo 才能确认，是否通知相应用户。
11. **`APST-open-source-20260915.tar.gz`（68M）**：GDrive 上是否已有副本？没有 rclone 无法确认；确认后删除。

## 9. 方法与限制

- 只使用 `du -x`（单文件系统）、`find`、`ls`、`stat`、`docker system df`/`docker images`（read-only）、`git status/log/ls-files`。唯一涉及 HDD 的重读取是 `zstd -dc | tar -t` 对 `results.tar.zst` 的前 120 秒流式读取（`nice -n 19 ionice -c3`，`timeout 120`），用于确认顶层目录。
- `/var/lib/docker`、`/home/{hw3090,ly,jiangjizhong}`、`/var/log/journal` 需要 root，无法精确测量。
- 没有运行 rclone，因此所有"GDrive 上有备份"的结论来自 AGENTS.md 和本地 rclone 日志，没有重新核对。
- 引用判断基于对 `SSM/{scripts,ssm_decode,docs,overleaf,results/analysis}` 和 `si10mad_fp/ws/experiments` 的 grep，不保证没有间接引用。
