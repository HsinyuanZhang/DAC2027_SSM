# hw3090 磁盘清理执行记录（2026-10-19）

执行依据：MZ 已批准的清理清单，对应审计文档 `SSM/docs/disk_audit_20261019.md`。每一项删除之前都做了验证，验证证据与命令记录如下。没有 commit，也没有 push。

清理前：`/` 为 914G 总量，740G 已用，128G 可用（86%）；`/mnt/data` 为 7.3T 总量，1.6T 已用，5.4T 可用（23%）。
清理后：`/` 为 914G 总量，689G 已用，179G 可用（80%）；`/mnt/data` 为 7.3T 总量，1.5T 已用，5.5T 可用（21%）。

## 通用前置检查

- `pgrep -a rclone` 开始时为空；整个过程中同一时刻只运行一个 rclone 任务（item 3 的 `rclone check`、item 6 的 `rclone copy` 与 `rclone check` 依次执行）。
- 当前运行中的 xinyuan 进程：`claude`、`pi` x2、`agy`、`zcode-cli`，以及 `run_p0solve.py select`（`TRITON_CACHE_DIR=/tmp/triton_p0solve`）。没有 vscode-server、cursor-server、nvm node 进程；`si10mad_fp` 的 python 进程在执行时已不在 `ps -u xinyuan` 中，但仍然没有碰 `si10mad_fp`。
- 对每个待删目录都运行了 `lsof +D <dir>`（和 `fuser -v`），没有进程占用（唯一的命中是我自己的 shell cwd）。
- 禁止触碰的路径（`/tmp/pi-*`、`/tmp/claude-1002`、`/tmp/pi-subagents-*`、`/tmp/triton_p0solve`、`/tmp/triton_streamf7`、`SSM/results/` 其余部分、`si10mad_fp`、`/mnt/data/datasets` 中的数据集本体、Docker）均未触碰。

## Item 1：NVMe A 类缓存与临时目录

验证：`lsof +D` 无占用；`find -mmin -360` 对被删的 `/tmp/triton_*`、`pytest-of-xinyuan`、`*_review` 没有 6 小时内修改的文件；`~/.triton/cache` 不被运行中的 job 使用（job 的 `TRITON_CACHE_DIR` 指向 `/tmp/triton_p0solve`）。

| 项 | 路径 | 大小 | 命令 | 结果 |
|---|---|---|---|---|
| pip cache | `~/.cache/pip` | 3.3G | `pip cache purge` | 删除 1001 个文件（3502.8 MB），剩余 36K |
| Triton 缓存 | `~/.triton/cache` | 1.3G | `rm -rf ~/.triton/cache` | 已删 |
| npm cache | `~/.npm/_cacache` | 0.62G | `npm cache clean --force` | 已删 |
| conda 缓存 | `~/miniconda3/pkgs`、index、tarballs | 约 4.4G（43 个 tarball 1.31 GB，29 个 package 3.06 GB） | `~/miniconda3/bin/conda clean --all -y` | envs 未动（`snn_cuda`、`spint` 仍在）；`spint` 的 torch 2.12.0+cu130 import 正常 |
| VS Code server 旧版本 | `~/.vscode-server/cli/servers/Stable-{07f806f9,2a59476c,4fe60c8b,645f29cc,7baac39e,8a7abeba,e4c7e7b1}...` | 约 4.4G（共 7 个，约 570-673M 每个） | `rm -rf`（逐个） | 保留 mtime 最新两个：`Stable-6d62b1d5...`（10-15 15:49）和 `Stable-16507d91...`（10-15 04:45）；`lru.json` 未动。没有 vscode 进程运行 |
| Cursor server 旧版本 | `~/.cursor-server/bin/linux-x64/{81fcf293...,e7a7e93f...}` | 约 0.78G | `rm -rf` | 保留最新两个 `e48ee610...`（06-13）和 `b887a26c...`（06-06） |
| cursor-agent 旧版本 | `~/.local/share/cursor-agent/versions/{2026.09.02-c22c1a3,2026.09.10-fd3934a}` | 约 1.05G | `rm -rf` | 保留 `2026.09.15-d2fe57e`（`~/.local/bin/agent` 与 `cursor-agent` symlink 指向它） |
| nvm node v24 | `~/.nvm/versions/node/v24.16.0` | 422M | `rm -rf` | `nvm current` 为 v26.1.0，`default -> v26.1.0`，没有进程使用 v24 |
| `/tmp/triton_*`（13 个） | `baselines`、`bl3`、`drift_1..3`、`hyb2`、`hybrid`、`hybrid2`、`pc3fix_gpu0_laneA..D`、`semgfix` | 约 2.7G | `rm -rf` | 最新 mtime 为 10-17，远超 6 小时；`/tmp/triton_p0solve` 与 `/tmp/triton_streamf7` 保留 |
| `/tmp/pytest-of-xinyuan` | | 225M | `rm -rf` | 已删 |
| `/tmp/*_review` | `bl_review`、`bl_review2`、`exp_review`、`expv2_review`、`hyb_review`、`m2_review`、`semg_review` | 约 90M | `rm -rf` | mtime 为 10-16/17，均已删 |
| `__pycache__`/`.pytest_cache`/`numba_cache*` | `~/Work_host` 下 690 个目录（排除 `.git`） | 约 131M | `xargs rm -rf` | 排除了 `iscas/np_mea_aligned_20261013`、`neuropixel_drift_comparison_20261009`、`neo_neighbor_20261016_v1`（si10mad_fp 引用），共 41 个目录。`SPINT/btransform_unified_v1/.../results/__pycache__` 下 2 个 `.pyc` 因权限被拒（只读存档），保留 |
| `SSM/.tools/triton_peft_gpu1` | | 150M | `rm -rf` | 验证：对 `SSM`（排除 `.git`、`results`、`.tools`）以及 `si10mad_fp` 做了 `grep -rIl triton_peft_gpu1`，唯一命中是审计文档本身；`git check-ignore` 确认 `.tools/` 被 git 忽略 |

Item 1 释放：NVMe 约 19G（`df` 已用 740G -> 721G，包含 item 1 全部）。

未处理（审批清单之外，故跳过）：`~/.vscode-server/data/CachedExtensionVSIXs`（235M）、`/var/log/journal`（需要 sudo）、`code-*` 启动器二进制（每个 33-35M）。

## Item 2：iscas 失败 attempt 残留

路径：`/home/xinyuan/Work_host/iscas/autonomous_canonical_noise300_20261014_v1/attempts/`（32G，mtime 10-14）。

验证：
- `lsof +D` 与 `fuser` 显示无进程占用。
- 读取了 `attempts/*.txt` 全部记录：`full_generation_attempt1_rss_exceeded.txt`（attempt 1 因 RSS 20.3GiB 超过 8GiB 目标而停止，没有最终输出）、`validation_attempt1_error.txt`（JSON 序列化 int64 错误）、`validation_attempt2_error.txt`（reverse index 映射错误）、`full_generation_attempt2_existing_partial_output.txt`（残留 partial 文件移入 attempts）。确认这些是失败 attempt。
- 成功的 `output/`（`canonical_independent_300s_10uVrms.f32` 24576000000 字节、`CANONICAL_NOISE_READY.json`、`IMMUTABLE_MANIFEST.json`）完整存在，`exits/generator.exit` 与 `supervisor.exit` 均为 `0`。
- 注意：`input/` 的 mtime 在今天变化，是因为 item 1 删除了其中的 `__pycache__`，不是其他进程写入。

删除的文件（均为稀疏 partial 二进制）：
- `attempts/white_independent_partial_attempt1.f32`（表观 24576000000 字节）
- `attempts/full_generation_attempt1_partial_output/canonical_independent_300s_10uVrms.f32`（表观 24576000000 字节）
- `attempts/full_generation_attempt1_partial_output/canonical_independent_300s_10uVrms.f32.forward.f64`（表观 49152344064 字节）

命令：`rm -f` 以上三个文件。保留全部小的 `.txt` 记录与空目录 `full_generation_attempt1_partial_output/`。`du -sh attempts`：32G -> 32K。

Item 2 释放：NVMe 32G（已用 721G -> 689G）。

## Item 3：`/mnt/data/DAC27_backup/SSM_old_20261015`

验证：
- `rclone check /mnt/data/DAC27_backup/SSM_old_20261015 gdrive:DAC27/SSM_old_20261015 --one-way --checkers 8 --tpslimit 8 --combined ...`：`0 differences found`，`3 matching files`（`= SSM_full.bundle`、`= tools.tar.zst`、`= results.tar.zst`）。检查耗时 1 分 35 秒，说明读取并比较了 hash。
- 另外独立核对 MD5：`rclone md5sum gdrive:DAC27/SSM_old_20261015` 与本地 `md5sum` 完全一致：`results.tar.zst` 1b85bdf277c49ba79d1185fe2cc2146e，`tools.tar.zst` bc45961d37e53adb9ee9c6c7011b8660，`SSM_full.bundle` a6fab470c3bf9cb0baa357a83f88f827。
- 当时没有其他 rclone 进程；`lsof +D` 无占用。

命令：`rm -rf /mnt/data/DAC27_backup/SSM_old_20261015`（14G）。

Item 3 释放：HDD 14G。注意：这个 tarball 现在只剩 GDrive 上的副本（已验证）。

## Item 4a：NinaPro DB6 zip

验证（Python `-I`，`zipfile.infolist()` 的 `file_size`/`CRC` 对比 `extracted/` 里文件的大小与 `zlib.crc32`）：
- 20 个 zip（`DB6_s{1..10}_{a,b}.zip`，共 21813252806 字节，约 21G）。
- 20 个 zip 里的全部 100 个数据文件（`.mat`；a 类每个 6 个，b 类每个 4 个）在 `extracted/<zip名>/` 下都存在，大小相同，CRC32 完全一致，合计 21813252806 字节的数据。
- 唯一没有出现在 `extracted/` 的 zip 成员是 macOS 元数据垃圾：每个 a 类 zip 内的 `__MACOSX/`、`__MACOSX/DB6_sN_a/`（空目录）、`DB6_sN_a/.DS_Store`（6148 B）和 `__MACOSX/DB6_sN_a/._.DS_Store`（120 B）。这些不是数据，我的脚本把空目录判成了 "missing_dir"，所以 a 类 zip 在严格模式下不是 100% "成员齐全"。我把这解释为不影响数据完整性（所有数据文件均 100% 通过），并按此删除。zip 可从 NinaPro 重新下载，`download_db6.sh` 与 `download.log` 保留。
- `lsof +D` 无占用。

命令：`rm -f DB6_s{1..10}_{a,b}.zip`（在 `/mnt/data/datasets/ninapro_db6/`）。`extracted/` 未动，仍为 21G。

Item 4a 释放：HDD 约 21G。

## Item 4b：`/mnt/data/work_host/envs/nomad_tf`（TF 2.15）

验证：
- `grep -rIn "nomad_tf" SSM/scripts SSM/ssm_decode`，过滤掉 `nomad_tf26`，只剩两行，是对脚本名 `run_baselines_nomad_tf.py` 的注释/docstring 引用，不是对 `envs/nomad_tf/` 的路径引用。对 `~/Work_host` 下 `.py/.sh/.md/.tex/.json` 搜索 `envs/nomad_tf/` 无命中（审计文档除外）。
- `ps` 中没有任何进程引用 `nomad_tf/`；`lsof +D` 无占用。
- `nomad_tf` 是基于 `~/miniconda3/envs/spint` 的 venv（`pyvenv.cfg`）；`nomad_tf26` 是独立的 conda env（自带 python3.8，没有指向 `nomad_tf` 的 symlink）。
- 删除后确认 `nomad_tf26` 正常：`import tensorflow` 版本 2.6.0。

命令：`rm -rf /mnt/data/work_host/envs/nomad_tf`（5.1G）。

Item 4b 释放：HDD 5.1G。

## Item 5：HDD 重复数据集 `backup_datasets/SNN_SpikeSorting`

路径与审计一致：备份 `/mnt/data/backup_datasets/SNN_SpikeSorting/{Hybridjanelia,MEArec,neuropixel,new_datasets,synthetic_spike_dataset}`；对照 `/mnt/data/work_host/SNN_SpikeSorting/` 同名目录（`~/SNN_SpikeSorting/*` 通过 symlink 指向 work_host 副本）。

验证：
- `lsof +D` 与 `fuser` 无占用。
- 对 5 个目录逐个执行 `rsync -rnc --itemize-changes --out-format='%i %n' <backup>/<d>/ <work_host>/<d>/`，5 个目录的 rc 都是 0，输出都是 0 行，即备份中每个文件在 work_host 副本中都存在且 checksum 一致。
- 独立复核：用 `find -printf '%P\t%s'` 比对相对路径加大小清单，`only-in-backup=0`（5 个目录全部）。文件数：Hybridjanelia 734、MEArec 42、neuropixel 54、new_datasets 134、synthetic_spike_dataset 378；work_host 侧每个目录多出 1 个文件，即 `BACKED_UP.md` 标记，没有其他差异。没有符号链接。
- 对最大文件（Hybridjanelia 的 `HYBRID_JANELIA_hybrid_static_siprobe_rec_64c_1200s_11_raw.mda`）两侧做 sha256，一致（75a1eaa5...c96d5）。
- 备份中没有"只存在于备份"的文件。

命令：在 `/mnt/data/backup_datasets/SNN_SpikeSorting` 下执行 `rm -rf Hybridjanelia MEArec neuropixel new_datasets synthetic_spike_dataset`。没有动 `figshare_25298815_kilosort4_sims`、`zenodo_1205233_*`、`.tmp_*`（唯一副本）。

Item 5 释放：HDD 约 92G（60G + 1.8G + 1.4G + 3.2G + 26G）。遗留：`/mnt/data/backup_datasets/README.md` 仍描述旧备份，现已过期（未修改）。

## Item 6：2026-10-15 之后的本地 results 备份（非破坏性）

选取方式：`SSM/results/` 下 `git ls-files` 计数为 0 的子目录，且其中最新文件 mtime 在 2026-10-15 之后。排除 `data_cache`、`p0solve_20261019`、`stream_f7_20261019`、`pi_logs`。

不纳入的目录及理由：
- `forward_eval_*`、`stage2_*`、`e1b_m2_20261007`、`labeled_calibration_20261007`：内容 mtime 在 10-05 至 10-07，只是在 10-15 20:31 从旧快照还原，已包含在已验证的 `SSM_old_20261015/results.tar.zst`（GDrive）中。
- `analysis`、`debug_audit`、`peft_round1`：已被 git 追踪；唯一例外是被忽略的 `results/analysis/db6_v2_20261016/summarize.log`，已纳入 bundle2。
- `expansion_20261016`：空目录。

Bundles（暂存于 `/mnt/data/DAC27_backup/results_20261019/`，`tar -I 'zstd -T4 -3'`）：
- `results_bundle1_baselines_20261016.tar.zst`：`results/baselines_20261016`（1339 个文件，1526396538 字节）
- `results_bundle2_misc_since_20261015.tar.zst`：`baselines_20261016_selsens`、`db6_20261015`、`db6_v2_20261016`、`drift_curve_20261016`、`expansion_round1`、`hybrid_identity_20261016`、`hybrid_identity_v2_20261017`、`hyser_20261015`、`m2_calib_20261016`、`population_calib_20261015`、`population_calib_p2_20261015`、`population_calib_p3_20261016`、`semg_calib_20261016` 以及 `analysis/db6_v2_20261016/summarize.log`（2831 个条目，258432119 字节）

打包验证：`zstd -tq` 通过；`tar -t` 条目数与源目录非目录项数一致（bundle1 1339 = 1339；bundle2 2831 = 2830 + 1）。`SHA256SUMS`：
- e5ee1a2ad904c86b14ad119e451cf9f8dfd0583d89a12bd7a42e235d85427514  results_bundle1_baselines_20261016.tar.zst
- c356ffe30ff674f9ef814cb963acba1c3e043549c2aa30ea3e595d31038af229  results_bundle2_misc_since_20261015.tar.zst

上传：`rclone copy /mnt/data/DAC27_backup/results_20261019 gdrive:DAC27/results_backup_20261019 --transfers 4 --checkers 8 --tpslimit 8 --drive-chunk-size 64M`（用时约 25 分钟，3/3 文件 100%）。日志：`/mnt/data/DAC27_backup/rclone_upload_results_20261019.log`。

上传验证：
- `rclone check ... gdrive:DAC27/results_backup_20261019`：`0 differences found`，`3 matching files`（含 `SHA256SUMS`）。
- `rclone md5sum` 与本地 `md5sum` 一致：bundle1 554ff7b6618e25e0d901b05f79407fbe，bundle2 111bf8a3a673e75e2e8624198acea626，SHA256SUMS a0c245849c3849757ada96486c1b6f97。本地 `sha256sum -c SHA256SUMS` 两个均为 OK。

清理：删除 HDD 暂存 bundles、`SHA256SUMS` 与空目录 `results_20261019/`（`rm -f` 加 `rmdir`）。原始 `SSM/results/` 全部保留，没有改动。GDrive 上保留 3 个文件。

Item 6 释放：HDD 0（暂存已删，净占用不变）。

## 汇总

| Item | NVMe 释放 | HDD 释放 |
|---|---|---|
| 1 | 约 19G | 0 |
| 2 | 32G | 0 |
| 3 | 0 | 14G |
| 4a | 0 | 约 21G |
| 4b | 0 | 5.1G |
| 5 | 0 | 约 92G |
| 6 | 0 | 0 |
| 合计 | 约 51G（`df` 可用 128G -> 179G） | 约 132G（`df` 可用 5.4T -> 5.5T，已用 1.6T -> 1.5T） |

最终 `df -h / /mnt/data`：

```
Filesystem                 Size  Used Avail Use% Mounted on
/dev/mapper/vgubuntu-root  914G  689G  179G  80% /
/dev/sda                   7.3T  1.5T  5.5T  21% /mnt/data
```

## 跳过/注意事项

- 没有任何一项因与审计不符而被整体跳过。
- Item 4a：a 类 zip 里的 macOS 元数据成员（`__MACOSX/`、`.DS_Store`、`._.DS_Store`）本来就没有被解压，已按"所有数据文件 100% 通过"处理，见上。
- Item 1 中 2 个只读 `.pyc` 因权限无法删除；若需要可后续处理。
- 审批清单之外未处理：`CachedExtensionVSIXs`、journal（需 sudo）、docker、iscas 大数据搬迁、`SSM/results` 其他目录、`/mnt/data/backup_datasets/README.md` 更新。
