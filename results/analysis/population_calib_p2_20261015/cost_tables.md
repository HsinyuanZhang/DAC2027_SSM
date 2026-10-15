### MAC / 一次校准 (B=8)
| quantity | 000688 P0 | 000688 P4 | HD-sEMG P0 | HD-sEMG P4 |
|---|---|---|---|---|
| acc MAC / valid bin (full) | 1,101 | 1,101 | 73,481 | 73,481 |
| acc MAC / trial | 242,352 | 242,352 | 91,854,071 | 91,854,071 |
| finalize MAC | 15,185 | 15,185 | 2,173,723 | 2,173,723 |
| calibrator forward MAC (all units) | 201,344 | 643,552 | 2,523,136 | 7,319,552 |
|   of which activation x activation | 0 | 13,312 | 0 | 131,072 |
|   of which fixed-weight (RRAM-mappable) | 201,344 | 630,240 | 2,523,136 | 7,188,480 |
| total MAC, one calibration (phi buffered) | 2,155,345 | 2,597,553 | 739,529,427 | 744,325,843 |
| total MAC, phi recomputed in pass 2 | 2,155,345 | 2,888,753 | 739,529,427 | 747,635,411 |
| fixed-weight share of all MAC (corrected) | 9.34% | 24.26% | 0.34% | 0.97% |
| old share (forward incl. act x act) | 9.34% | 24.78% | 0.34% | 0.98% |

### SRAM (KiB), P0, B=8; 表 int8 双缓冲, 描述量 16 bit
| case | option | acc width | accumulators (data+counters) | of which counters (B) | q_* trial scratch | lag delay lines | two-level staging | descriptors | table x2 | pool state | TOTAL | TOTAL + P4 phi buffer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000688 | fp32 | 32 | 6.5 | 68 | 0.3 | 0.1 | 0.0 | 5.4 | 2.0 | 0.0 | 14.4 | - |
| 000688 | perAdd_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 0.0 | 6.8 | - |
| 000688 | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 3.0 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 0.0 | 6.6 | - |
| 000688 | twoLevel_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 5.8 | 2.7 | 0.5 | 0.0 | 12.5 | - |
| HD-sEMG | fp32 | 32 | 313.2 | 68 | 6.0 | 2.5 | 0.0 | 80.0 | 32.0 | 0.0 | 434.8 | - |
| HD-sEMG | perAdd_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 210.6 | - |
| HD-sEMG | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 153.2 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 207.7 | - |
| HD-sEMG | twoLevel_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 301.1 | 40.0 | 8.0 | 0.0 | 511.8 | - |
| HD-sEMG | perAdd_cont28 | 28 | 273.1 | 34 | 5.3 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 329.9 | - |
| HD-sEMG | perAdd_int8code21 | 21 | 204.9 | 34 | 4.0 | 1.3 | 0.0 | 40.0 | 8.0 | 0.0 | 259.1 | - |

### SRAM (KiB), P4, B=8; 表 int8 双缓冲, 描述量 16 bit
| case | option | acc width | accumulators (data+counters) | of which counters (B) | q_* trial scratch | lag delay lines | two-level staging | descriptors | table x2 | pool state | TOTAL | TOTAL + P4 phi buffer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000688 | fp32 | 32 | 6.5 | 68 | 0.3 | 0.1 | 0.0 | 5.4 | 2.0 | 1.1 | 15.5 | 18.7 |
| 000688 | perAdd_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 1.1 | 7.8 | 11.1 |
| 000688 | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 3.0 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 1.1 | 7.7 | 10.9 |
| 000688 | twoLevel_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 5.8 | 2.7 | 0.5 | 1.1 | 13.6 | 16.8 |
| HD-sEMG | fp32 | 32 | 313.2 | 68 | 6.0 | 2.5 | 0.0 | 80.0 | 32.0 | 1.1 | 435.8 | 467.8 |
| HD-sEMG | perAdd_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 211.7 | 243.7 |
| HD-sEMG | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 153.2 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 208.8 | 240.8 |
| HD-sEMG | twoLevel_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 301.1 | 40.0 | 8.0 | 1.1 | 512.8 | 544.8 |
| HD-sEMG | perAdd_cont28 | 28 | 273.1 | 34 | 5.3 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 331.0 | 363.0 |
| HD-sEMG | perAdd_int8code21 | 21 | 204.9 | 34 | 4.0 | 1.3 | 0.0 | 40.0 | 8.0 | 1.1 | 260.1 | 292.1 |
