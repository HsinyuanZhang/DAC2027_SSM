# Colab runbook (DAC2027 SSM): GPU VM setup, parity check, job control

Recorded 2026-10-16 from the M2 population-calibration round (`results/analysis/m2_calib_20261016/README.md`);
revised the same day after the independent review (orchestrator/retry/loader fixes, see sections 1-3).
Session `dac27`: NVIDIA L4 24 GB, 12 vCPU, 52 GB RAM, Ubuntu 24.04, Python 3.13.15, driver 580.82 / CUDA 13.0,
about 1.54 CU/h.  Every command below was run exactly like this.

## 0. Preconditions (needs MZ)

* Drive mount: MZ runs `colab drivemount -s dac27` once per new session.  The project folder is then
  `/content/drive/MyDrive/DAC27` (shared from the rclone account).  Do not re-mount / recreate the session.
* Bundles live in `gdrive:DAC27/colab/` (uploaded by the host with rclone, one rclone job at a time) with a
  `SHA256SUMS` file: `code.tar.zst` (SSM repo at HEAD + `.tools/mamba_official` + einops/huggingface_hub, no triton),
  `data_falcon_m2.tar.zst` (contents: DANDI 000954, NOT M2, see step 3 in section 1), `ckpts.tar.zst`, `data_000688_subM.tar.zst`.
* Session life is about 12 h; make every job resumable and copy results to Drive often.

## 1. Environment recipe (exact)

```bash
# --- on the host: start/stop
colab usage                       # balance and rate; check at start, about every 30 min, and at the end
colab status -s dac27
colab stop -s dac27               # ALWAYS when done
```

On the VM (via `colab exec -s dac27 -f script.py`; all of the following are plain `subprocess.run` calls in such a script):

```bash
# 1. zstd (not preinstalled)
apt-get install -y -qq zstd       # if it fails: apt-get update -qq first

# 2. copy bundles from the Drive FUSE mount to local disk BEFORE extracting; verify hashes
mkdir -p /content/stage && cd /content/stage
cp /content/drive/MyDrive/DAC27/colab/{code,ckpts,data_falcon_m2,data_000688_subM}.tar.zst .
sha256sum *.zst                   # compare with /content/drive/MyDrive/DAC27/colab/SHA256SUMS

# 3. repo at the SAME absolute path as on the host (no code path changes)
mkdir -p /home/xinyuan/Work_host && cd /home/xinyuan/Work_host && tar -I zstd -xf /content/stage/code.tar.zst
cd /home/xinyuan/Work_host/SSM && tar -I zstd -xf /content/stage/ckpts.tar.zst      # paths inside are results/...
#    000688 data (tar paths are dandi_000688/sub-M/...), same absolute path as on the host
mkdir -p /home/xinyuan/Work_host/SPINT/sua_exploration/data
cd /home/xinyuan/Work_host/SPINT/sua_exploration/data && tar -I zstd -xf /content/stage/data_000688_subM.tar.zst
#    (the 000688 bundle also contains the FINAL sessions 0623/0625/0626: never read them)
#    DATASET IDENTITY: FALCON M2 (MonkeyN) is DANDI 000953 (15 GB of NWB).  The bundle named
#    data_falcon_m2.tar.zst actually contains DANDI 000954 (HumanPitt, 98 MB) under
#    /mnt/data/work_host/SPINT/SPINT-main/data/000954 and is NOT M2 (the name is misleading; do not use it as M2 data).
#    The M2 runs read the npz cache results/data_cache/m2/held_in/*.npz (inside ckpts.tar.zst) through
#    ssm_decode/m2_calib.load_m2_recording, so the NWB files are not needed.  That loader checks the cache
#    against results/analysis/m2_calib_20261016/m2_npz_manifest.json (pinned NWB sha256, data root and array digest;
#    the stored path must be a 000953 MonkeyN held-in file) and raises on any mismatch or missing metadata.
#    The manifest is part of the git tree, hence of code.tar.zst.  Re-hash the NWB where they exist with
#    `python scripts/m2_npz_manifest.py build --verify-nwb` (host only; the 7 NWB sha256 matched on 2026-10-16).
#    ssm_decode.multi_target.cached_recording would need the NWB (it re-hashes them on every load).

# 4. APST source (ssm_decode/data.py and dandi688_data.py import apst); 125 KB, not part of code.tar.zst
#    host: cd /home/xinyuan/Work_host/APST && tar --exclude=__pycache__ -czf /tmp/apst_src.tgz src
#    host: colab upload -s dac27 /tmp/apst_src.tgz /content/apst_src.tgz
mkdir -p /home/xinyuan/Work_host/APST && cd /home/xinyuan/Work_host/APST && tar -xzf /content/apst_src.tgz

# 5. python packages missing on the VM image (everything else is already there)
pip install pynwb==4.0.0 hdmf==6.1.0       # h5py 3.14.0 is preinstalled; pytest 8.4.2 is preinstalled

# 6. isolated triton for the official Mamba-3 kernels (the repo refuses a triton that is not under .tools/mamba_deps)
pip install --target /home/xinyuan/Work_host/SSM/.tools/mamba_deps triton==3.5.0
#    pip prints "torch 2.11.0+cu130 requires triton==3.6.0 ... incompatible": expected and harmless.  The VM's
#    system triton stays 3.6.0; the isolated 3.5.0 is picked up through PYTHONPATH.

# 7. git refuses the extracted official Mamba-3 clone (uid 1002 owner); compatibility_attempt calls `git rev-parse`
git config --global --add safe.directory /home/xinyuan/Work_host/SSM/.tools/mamba_official
```

Version table (VM): torch 2.11.0+cu130, triton 3.5.0 (isolated; system 3.6.0), numpy 2.1.3, pandas 2.2.3, scipy 1.16.3,
scikit-learn 1.6.1, einops 0.8.2, pynwb 4.0.0, hdmf 6.1.0, h5py 3.14.0, pytest 8.4.2.  Host (for reference):
torch 2.5.1 / py3.10 / triton 3.5.0 (mamba_deps), GPU RTX 3090.

Launcher environment for every Python process (never import torch/triton from the notebook kernel for jobs):

```bash
cd /home/xinyuan/Work_host/SSM
PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM python <script>
```

### Mamba-3 check (do this before any science)

`tests/test_mamba3_official.py` PASSES VACUOUSLY: it also accepts `status == unsupported_or_unavailable`, so a green
run of it says nothing about GPU/kernel availability.  Do NOT use it as the check.  The real check is
`compatibility_attempt('cuda:0')`, asserted:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=... python - <<'CHECK'
import json
from ssm_decode.mamba3_official import compatibility_attempt as c
r = c('cuda:0'); print(json.dumps(r, indent=1))
assert r['status'] == 'supported' and r['kernel_triton_matches_runtime'] is True, 'Mamba-3 NOT usable'
assert r['causal_future_perturb_max_abs'] == 0.0
CHECK
```

It must exit 0 and print `"status": "supported"`, `triton_version 3.5.0`, `kernel_triton_matches_runtime true`,
`causal_future_perturb_max_abs 0.0` (5.1 ms per forward on the L4).  With the git `safe.directory` step missing it silently
reports `unsupported_or_unavailable` (git exit status 128) and the assertion fails, as it should.

CPU-only alternative (no GPU needed, used for the review reproduction and for `scripts/run_m2_r1lr.py`): the plain-export
executor `ssm_decode/mamba3_cpu.py`, which reproduced the Colab rows to |dR2| <= 7e-5.

### Parity check (P1 `P0` rows, 000688 days / pool O / seed 0, B in {1, 8}, 2 draws)

```bash
S=/home/xinyuan/Work_host/SSM/results/population_calib_20261015
mkdir -p /content/parity/runs/days_O/p0/seed0
cp $S/runs/days_O/p0/seed0/{best.pt,done.json} /content/parity/runs/days_O/p0/seed0/
PC_OUT=/content/parity python scripts/run_population_calib.py evaluate --stratum days --pool O --variant p0 --tag p0 \
    --seed 0 --budgets 1 8 --draws 2 --device cuda:0
# compare /content/parity/runs/days_O/p0/seed0/eval_rows.jsonl with $S/runs/days_O/p0/seed0/eval_rows.jsonl on (target,B,draw)
```

`PC_OUT` redirects the rows so the stored ones are not skipped by the resume logic.  Result: see the M2 README (max |dR2| 6.2e-5).

## 2. Job control

* `colab exec -s dac27 -f x.py` returns when the cell returns: start long jobs with
  `subprocess.Popen("cd REPO && nohup bash -c 'ENV python script ...' >> /content/logs/NAME.log 2>&1 &", shell=True)`
  (use `setsid` for jobs that must survive a kernel restart) and poll with `tail`.
* Several GPU jobs run side by side on the L4 (4 concurrent processes were fine: 24 GB, 12 CPUs).
  `scripts/colab_orch.py` (upload to `/content/`; `python colab_orch.py N QUEUE.jsonl DONE.txt [options]`, started with
  `setsid nohup`) is a small chain orchestrator: it reads `{"name", "cmds"}` lines from the queue file (re-read every
  10 s, so it can be appended to), runs up to N chains concurrently and records finished chains in the DONE file given on
  the command line (it is honoured; the old version always wrote `/content/queue_done.txt`).  A chain that exits non-zero is
  appended to `DONE.txt.failed` (name, rc, time), logged to stdout, never retried by the same process, and does not block
  the exit.  Commands are passed to `bash -c` as one argument, so single quotes are fine now.  GPU throughput is the limit:
  the Mamba-3 backward (ui_reg) ran at about 33 ms/step alone and about 130 ms/step with three such jobs sharing the L4.
* **Never leave the VM running silently.**  The orchestrator now EXITS when every queue line is done or failed and nothing is
  running (after `--grace` 60 s, in which an appended line is still picked up), writes `/content/orch.finished` (JSON: done,
  failed) and optionally runs `--on-exit CMD`.  Guards: `--max-hours H` (stop launching after H hours), `--stop-file F`
  (default `/content/orch.stop`: no new chains, exit when the running ones finish), `--idle-exit S`.
  The host must still call `colab stop -s dac27` (the VM bills while assigned): poll `/content/orch.finished`
  (`colab ls`/`colab exec`) and stop; an untested VM-side option is `--on-exit "python -c 'from google.colab import runtime; runtime.unassign()'"`.
* **CU-check hook.**  `colab usage` only exists on the host, so the check is host-driven: every ~30 min run `colab usage`;
  if the balance is below the floor (the M2 round used a 40 CU cap per round), `colab exec` a file that does
  `open('/content/orch.stop','w').close()` (orchestrator stops launching) and then `colab stop -s dac27` once the running
  chains are done or can be resumed later (chains are resumable).  VM-side variant: `--hook-cmd CMD --hook-interval S`
  runs CMD periodically, e.g. a script that reads a balance file the host refreshes; a non-zero exit stops launching.
* **`colab_exec_retry.sh` and background launches.**  A retry may only re-run an exec file that is safe to repeat.  The
  wrapper therefore re-runs only on a NON-ZERO exit code, never on rc 0 (an rc 0 whose output still mentions
  "Connection was lost"/"TimeoutError" exits with 3: outcome unknown, check the VM instead of re-running), and a file whose
  first line starts with `# NONIDEMPOTENT` gets exactly one attempt.  Launch files should be made idempotent anyway:

  ```python
  import os, subprocess
  try:
      os.close(os.open("/content/.launched_orch", os.O_CREAT | os.O_EXCL))   # lock: a second run skips the launch
      subprocess.Popen("cd REPO && setsid nohup python scripts/colab_orch.py 4 /content/queue.jsonl /content/queue_done.txt >> /content/logs/orch.log 2>&1 &", shell=True)
  except FileExistsError:
      print("already launched")
  ```
* Triton re-autotunes the Mamba-3 backward kernel for every new sequence length.  Gradient code that feeds
  variable-length trials must pad to a fixed multiple of 64 bins (zeros at the END; the trunk is causal), otherwise the
  first epochs take minutes per trial length.
* `colab exec` sometimes fails with `RuntimeError: Connection was lost` / `TimeoutError` while background jobs load the VM.
  Background jobs are NOT affected (they are not children of the websocket).  Retry the exec (a retry loop with 6
  attempts and an 8 s sleep always worked, see `scripts/colab_exec_retry.sh`); `colab restart-kernel -s dac27` also worked
  (it kills jobs started WITHOUT `setsid`, such as an orchestrator started with plain `nohup`).
  `colab ls / download / upload` use another channel and keep working while exec hangs.
* A command that runs longer than 120 s in the foreground is moved to the background by the host tooling; use
  `--timeout N` for `colab exec` and keep each call short.
* Results: `tar -czf /content/out.tgz ...` on the VM, `cp` to `/content/drive/MyDrive/DAC27/colab/out/` (write to a `.tmp`
  name, then `mv`), and `colab download -s dac27 /content/out.tgz LOCAL` for the host.  Do not run rclone against Drive
  from the agent (one rclone job at a time; the host owns it).
* Tests on the VM: `python -m pytest tests/test_population_calib.py tests/test_m2_calib.py tests/test_colab_tools.py -q`
  (`test_mamba3_official.py` may be added but proves nothing about GPU support, see the Mamba-3 check; the M2 round had 39 tests green in
  35 s with it, before the review additions).
