# DANDI 000688 sub-M adapter inventory (2026-10-04)

## 1. Scope

1. This note documents the new data adapter `ssm_decode/dandi688_data.py`.
2. It covers subject `sub-M`, task `CO`, all 22 sessions on the local SSD tree.
3. Raw root is `/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M`.
4. Set `SSM_DANDI688_ROOT` to redirect to another tree, for example a merged CO+RT root.
5. The six RT sessions live on HDD and stay out of scope. The root resolution already accepts a merged tree.

## 2. Session inventory

Counts are after the M1 filter (`electrode_group_M1` plus `Primary Motor Cortex`, 96 canonical electrodes). Bins are 20 ms. Trials are trials-table rows. Rewarded is `result == "R"`. Rate is mean spikes per second per unit.

| Session | Date | Units | Bins | Trials | Rewarded | Rate (Hz) | Eval fraction |
|---|---|---|---|---|---|---|---|
| sub-M_ses-CO-20140203 | 2014-02-03 | 34 | 68509 | 245 | 182 | 6.59 | 0.75 |
| sub-M_ses-CO-20140217 | 2014-02-17 | 44 | 64275 | 255 | 208 | 6.75 | 0.77 |
| sub-M_ses-CO-20140218 | 2014-02-18 | 38 | 62031 | 259 | 225 | 8.64 | 0.85 |
| sub-M_ses-CO-20140303 | 2014-03-03 | 52 | 52723 | 233 | 209 | 7.69 | 0.87 |
| sub-M_ses-CO-20140304 | 2014-03-04 | 39 | 65674 | 235 | 203 | 10.81 | 0.75 |
| sub-M_ses-CO-20140306 | 2014-03-06 | 43 | 62930 | 264 | 219 | 8.51 | 0.80 |
| sub-M_ses-CO-20140307 | 2014-03-07 | 26 | 59764 | 254 | 216 | 9.74 | 0.75 |
| sub-M_ses-CO-20140626 | 2014-06-26 | 26 | 253616 | 1198 | 671 | 9.03 | 0.53 |
| sub-M_ses-CO-20140627 | 2014-06-27 | 25 | 260738 | 1250 | 609 | 9.50 | 0.45 |
| sub-M_ses-CO-20140929 | 2014-09-29 | 21 | 220017 | 805 | 733 | 10.93 | 0.82 |
| sub-M_ses-CO-20141203 | 2014-12-03 | 10 | 82531 | 223 | 194 | 17.41 | 0.76 |
| sub-M_ses-CO-20150511 | 2015-05-11 | 24 | 225292 | 621 | 536 | 8.27 | 0.57 |
| sub-M_ses-CO-20150512 | 2015-05-12 | 30 | 319008 | 563 | 443 | 5.78 | 0.59 |
| sub-M_ses-CO-20150610 | 2015-06-10 | 18 | 47713 | 208 | 190 | 10.06 | 0.87 |
| sub-M_ses-CO-20150611 | 2015-06-11 | 16 | 54771 | 215 | 196 | 9.01 | 0.84 |
| sub-M_ses-CO-20150612 | 2015-06-12 | 15 | 59492 | 206 | 175 | 9.21 | 0.86 |
| sub-M_ses-CO-20150615 | 2015-06-15 | 17 | 58349 | 205 | 174 | 9.11 | 0.84 |
| sub-M_ses-CO-20150616 | 2015-06-16 | 17 | 53811 | 211 | 183 | 9.27 | 0.82 |
| sub-M_ses-CO-20150617 | 2015-06-17 | 17 | 60097 | 223 | 175 | 9.24 | 0.77 |
| sub-M_ses-CO-20150623 | 2015-06-23 | 20 | 46706 | 207 | 183 | 11.68 | 0.86 |
| sub-M_ses-CO-20150625 | 2015-06-25 | 27 | 52044 | 219 | 174 | 10.44 | 0.78 |
| sub-M_ses-CO-20150626 | 2015-06-26 | 22 | 45268 | 208 | 184 | 12.04 | 0.85 |

Unit counts range from 10 to 52. This is the variable-per-session neural width that Track A needs.

## 3. Eval mask rule

The raw NWB files carry no evaluation mask. The adapter derives one.

1. A bin is eval-eligible if it belongs to a trials-table interval with `result == "R"`.
2. The bin center must also lie inside the `cursor_vel` timestamp support.
3. This keeps the interpolation from scoring fabricated zero velocity at the edges.
4. Inter-trial gaps and aborted, failed, or incomplete trials are excluded.
5. `trial_bounds` instead covers every trials-table row, any result, in table order.

## 4. Canonical plan

1. The default plan reproduces the frozen SPINT sub-M 2015 CO protocol.
2. Source pool is the six train sessions 20150511 through 20150615.
3. First targets are the two dev sessions 20150616 and 20150617.
4. The three final sessions 20150623, 20150625, and 20150626 stay withheld.
5. They are available on request through `dandi688_plan(..., include_final=True)`.
6. Source and target sets are checked for overlap and rejected on any shared session.
7. Protocol reference is `/mnt/data/work_host/SPINT/apst_final_week_20260916/dandi688_subm_2015_20260921/src/protocol.py` with schema `dandi688_subm_2015_co_v1`.

## 5. Load times

All reads were from the SSD tree with `pynwb` 2.8.2.

1. Full metadata scan of all 22 sessions took 95 s, about 4.3 s per session.
2. Full load of sub-M_ses-CO-20150612 (15 units, 59k bins) took 1.5 s.
3. Full load of sub-M_ses-CO-20150616 (17 units, 54k bins) took 2.0 s.
4. Full load of sub-M_ses-CO-20150512 (30 units, 319k bins, largest file at 362 MB) took 14.8 s.
5. No session needed the metadata-only fallback. Nothing came close to the 3 min limit.
6. The test suite `tests/test_dandi688_data.py` runs in about 8 s.

## 6. Cross-checks

1. sub-M_ses-CO-20150511 yields 225292 bins and 24 units. The SPINT prepared npz reports the same shape.
2. The final sessions yield 20, 27, and 22 units. The SPINT preparation receipt reports the same counts.
3. The 96-electrode canonical map matches the SPINT `canonical_electrode_keys` list of bank A, B, C times pin 1 to 32.

## 7. Anomalies

1. sub-M_ses-CO-20141203 has only 10 M1 units. It also has the highest mean rate at 17.41 Hz. Treat it as a low-width outlier.
2. sub-M_ses-CO-20140626 and sub-M_ses-CO-20140627 are marathon sessions of about 85 min with 1198 and 1250 trials. They contain many failed trials, 473 and 592. Their eval fraction drops to 0.53 and 0.45.
3. Cursor velocity covers at least 99.83 percent of every session span. The uncovered part is a short tail of 0.01 to 1.94 s after the last velocity sample. No mid-session velocity gaps were found.
4. Some sessions sort multiple units onto the same electrode. For example sub-M_ses-CO-20150612 has 15 units on 12 distinct electrodes. The `observed_mask` handles this by marking electrodes, not units.
5. The 2015 May sessions are long, 4506 to 6380 s, while the June sessions are near 1200 s. Session-length variance is large even inside the source pool.
6. The calendar gap between 2014-12-03 and 2015-05-11 is over four months. This is the longest inter-session gap in the roster and a natural drift stress test.
7. Trial result codes vary in proportion across sessions. Codes are R for rewarded, A for aborted, F for failed, and I for incomplete.
