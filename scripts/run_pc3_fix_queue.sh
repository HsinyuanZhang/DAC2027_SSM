#!/bin/bash
# P3 review-fix GPU queue.  Two lanes (<= 2 parallel jobs).  Resumable (every stage appends rows and skips
# finished keys).  Usage:  PC3_GPU=<physical gpu index> bash scripts/run_pc3_fix_queue.sh [laneA|laneB]
# The scripts address "cuda:0"; CUDA_VISIBLE_DEVICES maps the chosen physical GPU to it.
cd /home/xinyuan/Work_host/SSM
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${PC3_GPU:?set PC3_GPU to the physical GPU index you may use} \
  PC_OUT=/home/xinyuan/Work_host/SSM/results/population_calib_p3_20261016/fix \
  PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_pc3fix_gpu${PC3_GPU}_$1
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
LOG=$PC_OUT/logs; mkdir -p $LOG
R="$PY scripts/run_pc3_refine.py"
U="$PY scripts/run_pc3_uiref.py"
lane=${1:-laneA}

if [ "$lane" = laneA ]; then
  # --- item 1: like-for-like reference rule, selected on the SAME OOS episodes from the P0 start
  $U select --stratum month --set oos --sessions 20140307,20140627 > $LOG/uiref_select_month_oos_a.log 2>&1
  $U select --stratum month --set oos --sessions 20140929,20141203 > $LOG/uiref_select_month_oos_b.log 2>&1
  $U write --stratum month --tag oos  > $LOG/uiref_write_month_oos.log 2>&1
  $U write --stratum month --tag oosc > $LOG/uiref_write_month_oosc.log 2>&1
  $U eval --stratum month --tag oosc > $LOG/uiref_eval_month_oosc.log 2>&1
  $U eval --stratum month --tag oos  > $LOG/uiref_eval_month_oos.log 2>&1
  $U eval --stratum year  --tag oos --rule-stratum month > $LOG/uiref_eval_year_oos.log 2>&1
  # reproduction check of the P2 reference rule (tag p2repro) -> should match the P2 p0_ui_reg rows
  $U eval --stratum month --tag p2repro --rule-file results/population_calib_p2_20261015/uireg/selection_month.json > $LOG/uiref_eval_month_p2repro.log 2>&1
  # IS reference from the P0 start (2x2 table); month/year share the simulation
  $U select --stratum month --set is --sessions 20140203,20140217,20140218 > $LOG/uiref_select_month_is_a.log 2>&1
  $U select --stratum month --set is --sessions 20140303,20140304,20140306 > $LOG/uiref_select_month_is_b.log 2>&1
  $U write --stratum month --tag is0 > $LOG/uiref_write_month_is0.log 2>&1
  $U eval --stratum month --tag is0 > $LOG/uiref_eval_month_is0.log 2>&1
  $U eval --stratum year  --tag is0 --rule-stratum month > $LOG/uiref_eval_year_is0.log 2>&1
fi

if [ "$lane" = laneB ]; then
  # --- item 2: causal month.  r12 rows for the single causal session (0307) are the identical episodes already
  #     simulated (seeded into fix/select/r12_month_oosc.jsonl); R3 selection and all target rows are new.
  $R select-r3 --stratum month --set oosc > $LOG/select-r3_month_oosc.log 2>&1
  $R eval --stratum month --set oosc > $LOG/eval_month_oosc.log 2>&1
  $R ablate --stratum month --set oosc > $LOG/ablate_month_oosc.log 2>&1
  # --- item 10: diagnostics with the realised update (original headline rules; month causal rule)
  $R diag --stratum month --set oosc > $LOG/diag_month_oosc.log 2>&1
  $R diag --stratum year --set oos > $LOG/diag_year_oos.log 2>&1
  $R diag --stratum days --set is > $LOG/diag_days_is.log 2>&1
  # --- item 3: capped replay (rules/lambdas selected without cap are reused: --sel-variant base)
  $R eval --stratum month --set oosc --variant cap256 --sel-variant base > $LOG/eval_month_oosc_cap256.log 2>&1
  $R eval --stratum year  --set oos  --variant cap256 --sel-variant base > $LOG/eval_year_oos_cap256.log 2>&1
  # --- item 6: contiguous-stream replay (own source selection, R1s not evaluated)
  $R select --stratum month --set oosc --variant contig > $LOG/select_month_oosc_contig.log 2>&1
  $R select-r3 --stratum month --set oosc --variant contig > $LOG/select-r3_month_oosc_contig.log 2>&1
  $R eval --stratum month --set oosc --variant contig > $LOG/eval_month_oosc_contig.log 2>&1
  $R select --stratum year --set oos --variant contig > $LOG/select_year_oos_contig.log 2>&1
  $R select-r3 --stratum year --set oos --variant contig > $LOG/select-r3_year_oos_contig.log 2>&1
  $R eval --stratum year --set oos --variant contig > $LOG/eval_year_oos_contig.log 2>&1
  # --- item 10: extended (L, S) grid on source selection (R2 only), then target evaluation of the ext rule
  $R jacobian --stratum month --lmax 64 > $LOG/jac_month_l64.log 2>&1
  cp $PC_OUT/jacobian_month_l64.pt $PC_OUT/jacobian_year_l64.pt     # month and year share trunk and sources
  $R select --stratum month --set oosc --variant ext > $LOG/select_month_oosc_ext.log 2>&1
  $R eval --stratum month --set oosc --variant ext > $LOG/eval_month_oosc_ext.log 2>&1
  $R select --stratum year --set oos --variant ext > $LOG/select_year_oos_ext.log 2>&1
  $R eval --stratum year --set oos --variant ext > $LOG/eval_year_oos_ext.log 2>&1
fi
