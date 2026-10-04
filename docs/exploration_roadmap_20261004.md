# Exploration Roadmap: CIM-Aware SSM Decoder with Partial-Label Association Profiles

Date: 2026-10-04. Status: plan. A step is complete only if this document gives its result and its source file.

## 0. Purpose and Scope

This roadmap defines the next research steps for the SSM cross-session decoder. The target is a compute-in-memory (CIM) system study. It is not a circuit plan or a tape-out plan.

The roadmap has four tracks:

1. Track A: build association profiles (AP) with matrix operations and a small number of labels.
2. Track B: adapt the SSM with state-offset methods.
3. Track C: improve the SSM model and its training contract.
4. Track D: design the system for CIM and ASIC limits.

Each track gives the method, the procedure, the measurements, a decision gate, and the references. Section 7 gives the phase order. Section 9 lists all references.

### 0.1 Terms

| Term | Meaning in this document |
|---|---|
| Unit | One source of spikes: one sorted neuron or one electrode channel. |
| AP | Association profile. A per-unit vector that describes how one unit relates to a context signal. |
| Context signal | The reference signal for the AP: behavior labels, or latent factors of the decoder. |
| CIM | Compute-in-memory. Here, RRAM crossbar arrays that do matrix-vector products. |
| 2T2R | A differential RRAM pair. One pair stores one signed digit in {-1, 0, +1}. |
| PE | Processing element. A digital, reconfigurable unit near SRAM. |
| LTI | Linear time-invariant. The recurrence coefficients do not change with the input. |
| Selective SSM | An SSM with input-dependent dt, A, B, and C. Mamba is selective. |
| Labeled trial | A calibration trial with behavior labels. |
| Unlabeled data | Neural data with no behavior labels. |
| srcval R² | R² on source-session validation data. This data has no cross-session shift. |
| Target R² | R² on query data of a new session. |
| PTQ, QAT | Post-training quantization, quantization-aware training. |
| UI | Unit identification. POSSM learns new unit embeddings with gradients. |

## 1. Starting Point

These items are results from this repository. Each item names its source.

1. Input normalization gives the largest single gain. A causal EMA z-score raises source-only target R² from 0.386 to 0.699 (M1) and from −0.006 to 0.266 (M2). This needs no training. Source: `docs/ssm_cross_session_audit_20261003.md`.
2. The round-1 PEFT comparison did not converge. In 31 of 48 adaptation fits, the best step was step 850 or later.
3. Rank-4 LoRA cannot express a per-channel gain change, because that change is full rank.
4. Mamba-3 256×4 does not fit a 128 KB SRAM limit. Its recurrent state needs approximately 202 KB at INT24. Mamba-3 128×2 needs approximately 67 KB in total.
5. 2T2R PTQ of Mamba-3 128×2, measured with srcval R²: ternary FFN and ternary B/C rows, with 8 bits elsewhere, lose 0.012 (M1) and 0.010 (M2). All-ternary loses 0.036 (M1) and 0.048 (M2). Source: `results/analysis/rram_quant_20261003/README.md`.
6. Target R² hides quantization loss. Random weight noise with the norm of the ternary error raised M2 target R² by 0.047. Thus, use srcval R² for quantization decisions.
7. In Mamba-3 128×2, static-weight matrix products are approximately 86% of all multiply-accumulate operations. The selective state update and the state readout are approximately 9%.
8. The APST AP (T4) is a per-unit least-squares fit of the movement-window firing rate on [1, cos θ, sin θ]. It needs center-out target angles. Source: `APST/src/apst/dandi_subm/carrier.py`.

## 2. Design Principles

Apply these five principles to all tracks.

1. Put products of a static weight and a dynamic input in RRAM. Put products of two dynamic values in the digital PE.
2. Put all session-specific operators at the array boundaries. Keep session-invariant operators inside the arrays.
3. Do not use on-chip backpropagation for target adaptation. Use accumulation, small matrix solves, and table writes.
4. Compare quantization choices with srcval R². Compare adaptation methods with target R² on many target sessions.
5. Put only hardware-legal adaptation methods in the deployable set. Full fine-tuning is a reference only, because it rewrites RRAM.

The table below lists the boundary operators for principle 2.

| Boundary operator | Location | Drift that it can absorb |
|---|---|---|
| Row gain and row offset | Input DAC side | Per-channel firing-rate gain and offset |
| Unit table (K×N read-in) | SRAM | Unit identity change, new units, lost units |
| Column gain and column offset | After the ADC | RRAM column conductance drift |
| Norm after an array | Digital PE | Global conductance drift of one array |
| State offset | Registers | Latent mean shift |
| Output ridge (D×D) | Digital PE | Output scale and output bias |

The device-drift entries in the last column are hypotheses. Track D tests them.

## 3. Track A: Matrix-Based AP with Partial Labels

### 3.1 Problem

A new session gives a small labeled set L and a large unlabeled set U. The method must make a K×N read-in table for the frozen decoder. The method can use only these operations:

1. Streaming outer-product accumulation.
2. Small solves (K×K or D×D, with K ≤ 32).
3. One table write per session.

Label-free methods can recover a latent subspace. They cannot recover the orientation of that subspace (the frame). The main hypothesis of Track A follows. A small number of labels can fix the frame, and unlabeled data can fix the per-unit table. The labels then estimate approximately K² numbers, not N×K numbers.

### 3.2 Notation

| Symbol | Shape | Meaning |
|---|---|---|
| x_t | N | Normalized spike counts of N units at bin t |
| y_t | D | Behavior label at bin t (only in L) |
| φ_t | K | Context features at bin t |
| H | N×K | Profile matrix. Row h_i is the AP of unit i. |
| ψ_i | scalar | Residual noise variance of unit i |
| A | K×K | Shared posterior matrix |
| T | K×K | Frame matrix |
| B_src | D×K | Source behavior head (see C8) |
| W_lift | d×K | Session-invariant lift into the backbone width d |

The encoding model is x_t = H φ_t + b + ε_t, with a diagonal noise covariance Ψ = diag(ψ_i).

### 3.3 Accumulators

All accumulators are sums of outer products. Each accumulator updates once per bin.

| Accumulator | Shape | Data | Labels needed |
|---|---|---|---|
| Unit mean and unit variance | N, N | L and U | No |
| Unit covariance Σ x_t x_tᵀ (for FA) | N×N | L and U | No |
| S_φφ = Σ φ_t φ_tᵀ | K×K | L, or L and U | Only if φ comes from labels |
| S_φx = Σ φ_t x_tᵀ | K×N | L, or L and U | Only if φ comes from labels |
| S_zy = Σ ẑ_t y_tᵀ | K×D | L only | Yes |

A digital near-memory unit keeps these sums. The same unit can do the SSM state update (D4).

### 3.4 Method Family

Each method below has a closed form. Each method is a sequence of matrix products and small solves.

**A-M0. Supervised AP with a general behavior basis (reference).**

1. Set φ_t = f(y_t). Use behavior features with time lags.
2. Compute Hᵀ = (S_φφ + λI)⁻¹ S_φx on L only.
3. Map each profile to a table row with a shared map g, as APST does with its MLP.

This method generalizes T4 to continuous behavior. Step 2 is the same calculation as the observation matrix of a ReFIT Kalman decoder.

**A-M1. Label-free unit table, labeled frame (main method).**

1. Fit a factor-analysis (FA) model with K factors on L and U with EM. This gives H and Ψ. FA defines the latent only up to a rotation.
2. Compute the posterior read-in W_FA = A Hᵀ Ψ⁻¹, with A = (I + Hᵀ Ψ⁻¹ H)⁻¹.
3. Compute the latent ẑ_t = W_FA x_t on L. Fit M = argmin Σ ||y_t − M ẑ_t||² with a D×D-sized ridge solve.
4. Find the orthogonal T that minimizes ||B_src T − M||. Use the SVD of B_srcᵀ M = U S Vᵀ. Then T = U Vᵀ.
5. When D < K, labels fix only part of T. Add a term that makes T ẑ_t follow the source latent dynamics of the frozen SSM. This idea is similar to NoMAD.
6. Write the table W = T W_FA (K×N) to SRAM.

Step 4 is exact because ||B_src T||² is constant for an orthogonal T. Step 4 needs one K×K SVD. C8 makes the source latent white, so that an orthogonal T can model the frame.

**A-M2. Shrinkage to a source population prior.**

1. Estimate the mean profile H̄₀ and the profile covariance Σ₀ over all source units.
2. Use them as a ridge prior: Hᵀ = (S_φφ + λΣ₀⁻¹)⁻¹(S_φx + λΣ₀⁻¹H̄₀ᵀ).

This is an empirical-Bayes estimate. It can help when L or U is very small.

**A-M3. Semi-supervised noise estimate.**

1. Estimate ψ_i from unlabeled data: ψ_i = Var_U(x_i) − h_i Σ_φ h_iᵀ.
2. Use ψ_i in the posterior read-in (A-M1, step 2).

The term 1/ψ_i gives each unit a reliability weight. This can explain why T4 contains the modulation depth and the baseline rate.

**A-M4. Pseudo-label refinement.**

1. Use the decoder outputs on U as weighted pseudo-labels.
2. Repeat the frame fit a maximum of three times.
3. Check each repetition on held-out labeled trials.
4. Stop if the held-out error increases.

Pseudo-labels can cause confirmation bias. Thus, A-M4 is an option, not the main method.

**A-M5. Associative prototype read-in (CIM-native).**

1. Store K_p prototype profiles M_p (K_p×K) and prototype embeddings E (K_p×K) from source units in RRAM.
2. For unit i, use the aligned profile T h_i from A-M1, or the behavior profile from A-M0.
3. Compute s_i = softmax(β M_p T h_i). One crossbar read gives all K_p similarities.
4. Set the table row of unit i to Eᵀ s_i.

A-M5 needs no matrix solve for the table rows.

### 3.5 Read-in Use at Inference

The decoder input is z_t = W_lift (W x_t) + c.

- W (K×N) is the session table in SRAM.
- W_lift (d×K) and the backbone are session-invariant. They stay in RRAM.
- With event-driven input, each spike adds one column of W to a K-dim accumulator.

This form is the same as a POSSM unit-embedding lookup. POSSM learns the table with gradients. Track A computes the table in closed form.

### 3.6 Procedure

1. Use DANDI 000688 as the primary dataset. It contains sorted units, a variable unit count, CO and RT tasks, and long time gaps.
2. Use O'Doherty 2017 and FALCON M1, M2, and H1 as secondary datasets.
3. Pretrain with a per-session K×N read-in, a shared W_lift, and a linear behavior head B_src (C8).
4. For each target session, set the labeled budget to 0, 2, 4, 8, 16, and 33 trials. Keep the unlabeled pool constant.
5. Compare A-M0 to A-M5 with these baselines:
   - Channel affine plus output ridge (fixed-channel data only).
   - UI with gradients on the same K×N table and the same labels.
   - APST T4 (CO sessions only).
   - Label-free FA plus Procrustes, as in Degenhart 2020 (fixed-channel data only).
6. Record the target R² of each target session. Record paired differences to the baselines.
7. Record the label-efficiency curve of each method.

### 3.7 Decision Gate

Continue Track A if A-M1 or A-M3 with 8 or fewer labeled trials matches A-M0 with 33 labeled trials on most target sessions. If not, find whether the frame T or the unit table H causes the gap.

### 3.8 Risks

1. FA on few bins can give an unstable H. Use A-M2 shrinkage.
2. The frame T can absorb errors that belong to H. Check H with held-out unit prediction: predict one unit from the other units.
3. FALCON has only 3 to 6 source sessions. This is too few for priors and prototypes. Use 000688 for these parts.

## 4. Track B: State-Offset Methods

### 4.1 Background

State-offset Tuning (Kang et al., ACL 2025) adds a learned offset h' to the state at readout: y_t = C_t(h_t + h'). The offset does not enter the recurrence. Thus, its effect does not decay over time. The same paper compares initial-state tuning and prompt-based methods.

These related methods also adapt an SSM with few parameters:

- BitFit (bias-only tuning).
- Prefix tuning.
- SDLoRA (sparse dimension tuning for SSMs).
- MambaPEFT.
- Memba (membrane-state adapter).

The round-1 port in this repository gave target R² 0.266 ± 0.011 (M2) and 0.555 (M1). The port used the rotated kernel frame. The port also trained a root LoRA. Thus, the round-1 result does not isolate the offset effect.

### 4.2 B1. Fix the Port

1. Use the original state frame: y += silu(z) ⊙ (C_t + b_C)ᵀ h'. Do not apply the RoPE rotation to C in the offset term.
2. Remove the root LoRA from this method. Use the channel affine from C1 instead.
3. Record this limit: in an LTI model with a static C, the offset term is a constant. It is then equal to an output bias.

The offset has its own effect only with an input-dependent C or with gating.

### 4.3 B2. Offset Basis from Source Sessions

1. During pretraining, give each source session its own offset in each layer.
2. After pretraining, stack the source offsets and do an SVD. Keep r basis vectors U (r ≤ 8).
3. For a target session, fit only the r coefficients v. The offset is h' = U v.

### 4.4 B3. Closed-Form Offset Fit with Few Labels

The offset enters only the readout. For the last layer, the path from h' to the output is nearly linear. Only the final LayerNorm is nonlinear.

1. Linearize the output at v = 0: ŷ_t(v) ≈ ŷ_t + J_t v. J_t is a D×r matrix.
2. Compute J_t from (C_tᵀ U), the fixed linear output path, and the LayerNorm Jacobian.
3. Solve v = (Σ J_tᵀ J_t + λI)⁻¹ Σ J_tᵀ (y_t − ŷ_t) over the labeled bins. This is one Gauss-Newton step.
4. Compare the result with a gradient fit of the same v.

For offsets in deeper layers, use an off-chip gradient fit.

### 4.5 B4. Offset from AP Statistics

1. Compute a session summary m_s from the Track A accumulators, for example the mean latent and the mean profile.
2. In source pretraining, train a small map G so that v = G m_s gives a good offset.
3. Apply G to the target summary. This needs no target gradient.

G is static. It can stay in RRAM.

### 4.6 B5. Label-Free Offset (Reference Only)

Choose v so that the target latent mean matches the source latent mean. Solve this as least squares. Use it only as a reference, because the project focus is partial labels.

### 4.7 B6. Online Offset

1. During use, update v slowly with an LMS rule.
2. Use labels when they are available.
3. Use the latent-mean term when no labels are available.
4. Compare the result with the causal EMA input normalization.

The audit found within-session drift. In M1, query R² in later quartiles was 0.09 to 0.14 lower than in the first quartile.

### 4.8 B7. Hardware View

- h' or v is a small register set.
- The extra readout per head is (C_tᵀ U) v. This is r dot products per head per step.
- If C4 fuses the readout into RRAM (LTI case), the offset becomes a column bias. The same bias can also cancel ADC column offsets.

### 4.9 Decision Gate

Keep a state-offset method only if it adds a gain above the noise floor on top of the Track A read-in, on most target sessions. Use the same labeled budget for all methods.

## 5. Track C: SSM Improvements

### 5.1 C1. Contract Fixes

1. Normalize each source session with its own statistics.
2. Implement normalization as a row gain on spike counts. Fold the offset term into the output bias. This keeps the input sparse.
3. Keep two target protocols separate: support-only z-score, and causal EMA z-score.
4. Sweep the learning rate for each method family. The best step must come before 80% of the step budget.
5. Evaluate on many target sessions.

### 5.2 C2. Continuous Stateful Training and Streaming Evaluation

The official Mamba-3 forward sets Input_States to None. The official step() needs CuteDSL, and this system does not have CuteDSL.

1. Call mamba3_siso_combined directly with Input_States and return_final_states.
2. Train on recording-continuous chunks. Pass the final state of each chunk to the next chunk.
3. Write a PyTorch streaming step() reference.
4. Test that the streaming output equals the chunked output.

### 5.3 C3. Model Size

1. Use Mamba-3 128×2 as the main candidate.
2. Test the state size N = 8 and N = 16.
3. Keep small S4D and osc models as parallel candidates. They already have a step() reference and a fixed-point model.

### 5.4 C4. Selective versus LTI (Main CIM Axis)

Selectivity moves computation out of the RRAM arrays. LTI moves computation back into the arrays.

Test these variants at equal size:

1. Full Mamba-3.
2. Fixed trap (0.5).
3. Fixed dt per head.
4. Data-independent angles from a phase accumulator.
5. Full LTI.

In the LTI case, fold B into the input projection. Fold C into the next projection: W·C·h = (W·C)·h. The state vector then becomes the input of one RRAM array.

### 5.5 C5. Complex versus Real State

1. Compare a real diagonal state and a complex (oscillatory) state at equal state memory.
2. If the complex state helps, test the decay-only form.

In the decay-only form, the state memory does only a shared scalar decay. A phase accumulator rotates B and C.

### 5.6 C6. Norm Type and Location

1. Compare RMSNorm and LayerNorm.
2. Put a norm after each RRAM block where the absolute scale has no meaning. This norm can cancel global conductance drift.

Evidence: the B/C rows have an RMSNorm after them. Ternary PTQ of the B/C rows changed srcval R² by −0.001 to +0.008.

### 5.7 C7. Activation

1. Replace SiLU with a piecewise-linear SiLU, or with ReLU in the FFN.
2. Retrain after the change.

SiLU is approximately 1,024 of the 1,161 nonlinear operations per step in Mamba-3 128×2.

### 5.8 C8. AP-Aware Pretraining

1. Use a per-session K×N read-in and a shared W_lift.
2. Add a linear behavior head B_src on the K-dim latent. Track A uses it to fit the frame.
3. Make the K-dim latent white (identity covariance) with a penalty term. A-M1 needs this property.
4. Use episodic training. In each step, compute the AP from k trials of one session. Then decode other trials of the same session.
5. Add channel gain augmentation (log-normal, σ ≈ 0.3) and channel dropout.
6. Pretrain on 000688, O'Doherty, and FALCON held-in data.

### 5.9 C9. Hardware-Aware Training

1. Use QAT with 2T2R slices: ternary FFN, ternary B/C rows, and multi-slice in_zx and out_proj.
2. Add device variation (σ = 5% to 10%) during training.
3. Add ADC partial-sum quantization (D3).
4. Use INT8 activations, INT16 or INT24 state, and Q1.14 coefficients.

## 6. Track D: CIM/ASIC-Aware System Design

### 6.1 D1. Operator Partition

| Operator | Location | Reason |
|---|---|---|
| Input projection rows z, x, B, C | RRAM | Static weight, high reuse |
| SSM output projection, FFN | RRAM | Static weight, high reuse |
| W_lift, AP prototypes (A-M5), map G (B4) | RRAM | Static and session-invariant |
| dt, A, trap, and angle rows | Digital PE, or remove them (C4) | Errors pass through exp and softplus |
| Selective state update and readout | Digital PE and state SRAM | Product of two dynamic values |
| Nonlinear functions and norms | Digital PE (LUT or PWL) | Not matrix products |
| Session table W, row gains, offsets | SRAM or registers | Rewritten for each session |
| AP accumulators | Digital near-memory unit | Outer-product sums with frequent writes |

### 6.2 D2. Boundary Operators

Use the table in Section 2. Track A, Track B, and C1 supply the session-specific values.

### 6.3 D3. Mixed-Precision Slices and ADC Model

1. Add ADC partial-sum quantization to `results/analysis/rram_quant_20261003/rram_ptq.py`.
2. Use bit-serial INT8 inputs.
3. Set the ADC resolution to 3, 4, 5, and 6 bits.
4. Set the rows per conversion to 32, 64, and 128.
5. Measure srcval R² for each setting.

### 6.4 D4. Shared Outer-Product Engine

The AP accumulators and the SSM state update are both outer-product sums. Design one digital unit for both tasks. The calibration mode uses it for some minutes. The inference mode uses it at each 20 ms step.

### 6.5 D5. Exploratory CIM Operators

1. One-step in-memory regression for the small K×K solves (Sun and Ielmini, PNAS 2019 and Science Advances 2020).
2. Analog CAM or crossbar similarity search for the A-M5 prototypes (Li et al., Nature Communications 2020).
3. In-situ outer-product accumulation in RRAM. Write endurance is the main limit. Use it for calibration only.

### 6.6 D6. Joint Drift Experiment

1. Inject neural drift: per-channel gain and offset, unit loss, new units, and latent rotation.
2. Inject device drift: per-row and per-column conductance change, global conductance change, and ADC offset.
3. Calibrate the boundary operators in two ways: one joint calibration, and two separate calibrations.
4. Measure target R² and srcval R².

Accept the joint-compensation claim only if the joint calibration is equal to or better than the separate calibrations under combined drift.

### 6.7 D7. Golden-Model Chain (Later)

1. Float streaming reference.
2. Fixed-point bit-exact reference.
3. RRAM behavior model with slices, variation, and ADC.
4. RTL for the digital PE, with vector comparison.

## 7. Phase Order

| Phase | Work | Exit condition |
|---|---|---|
| 0 | C1, C2, C3. Get 000688 and O'Doherty data. Build a multi-target evaluation. | Streaming and chunked outputs agree. The new normalization contract runs on all datasets. |
| 1 | Track A on a frozen backbone: A-M0 to A-M3. | Decision gate 3.7. |
| 2 | Track B: B1 to B4. C4 and C5 ablations. | Decision gate 4.9. C4 gives the accuracy cost of each removed feature. |
| 3 | C8, C9, D3, A-M4, and A-M5. | The mixed-precision model with the ADC model loses at most 0.02 srcval R². |
| 4 | D5 and D6. | Data accepts or rejects the joint-compensation claim. |

## 8. Stop Rules

1. Stop a method if it does not exceed the noise floor on most target sessions after its learning-rate sweep.
2. Do not use target R² from one target session to compare quantization settings.
3. Do not report an energy number or an area number before an energy model with measured macro parameters exists.
4. Do not call a method novel before a dated literature search.

## 9. References

Check each reference before you cite it in a paper. Items with [check] need a check of authors, venue, or identifier.

### 9.1 SSM and Sequence Models

1. Gu et al. S4D: "On the Parameterization and Initialization of Diagonal State Space Models." NeurIPS 2022.
2. Smith, Warrington, Linderman. S5: "Simplified State Space Layers for Sequence Modeling." ICLR 2023.
3. Orvieto et al. LRU: "Resurrecting Recurrent Neural Networks for Long Sequences." ICML 2023.
4. Dao, Gu. Mamba-2. arXiv:2405.21060. ICML 2024.
5. Mamba-3. arXiv:2603.15569. ICLR 2026.
6. Rusch, Rus. LinOSS. arXiv:2410.03943. ICLR 2025.
7. Yang, Kautz, Hatamizadeh. Gated DeltaNet. arXiv:2412.06464. ICLR 2025.
8. Wang, Shi, Fox. "Test-time regression: a unifying framework for designing sequence models with associative memory." 2025. [check]

### 9.2 PEFT and State-Offset Methods

1. Kang et al. State-offset Tuning. arXiv:2503.03499. ACL 2025. [check authors]
2. Galim et al. SDLoRA. arXiv:2410.09016. ICML 2025.
3. Yoshimura et al. MambaPEFT. arXiv:2411.03855. ICLR 2025.
4. Memba. arXiv:2506.18184. ICLR 2026.
5. Hu et al. LoRA. ICLR 2022.
6. Liu et al. DoRA. ICML 2024.
7. Liu et al. (IA)³: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning." NeurIPS 2022.
8. Ben Zaken, Ravfogel, Goldberg. BitFit. ACL 2022.
9. Li, Liang. Prefix-Tuning. ACL 2021.

### 9.3 Neural Decoding and Cross-Session Alignment

1. POSSM. arXiv:2506.05320. NeurIPS 2025.
2. Zhang et al. APST. arXiv:2609.39080.
3. Azabou et al. POYO: "A Unified, Scalable Framework for Neural Population Decoding." NeurIPS 2023.
4. Ye et al. NDT2: "Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity." NeurIPS 2023.
5. Karpowicz et al. FALCON benchmark. NeurIPS 2024 Datasets and Benchmarks.
6. Degenhart et al. "Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity." Nature Biomedical Engineering 2020.
7. Karpowicz et al. NoMAD: "Stabilizing brain-computer interfaces through alignment of latent dynamics." [check venue]
8. Farshchian et al. ADAN: "Adversarial Domain Adaptation for Stable Brain-Machine Interfaces." ICLR 2019.
9. Gilja et al. ReFIT: "A high-performance neural prosthesis enabled by control algorithm design." Nature Neuroscience 2012.
10. Garnelo et al. "Conditional Neural Processes." ICML 2018.
11. SPINT: spatial permutation-invariant neural transformer for intracortical motor decoding. 2025. [check]
12. O'Doherty et al. 2017 reaching dataset. Zenodo, DOI 10.5281/zenodo.583331.
13. DANDI 000688, version 0.250122.1735 (monkeys C and M, M1 and PMd).

### 9.4 CIM and Hardware

1. Zhang et al. (Wei D. Lu group). CIM implementation of SSMs for event sequences. Nature Communications 2026. DOI 10.1038/s41467-025-68227-w.
2. Siegel, Yang, Strachan. IMSSA. arXiv:2412.20215.
3. Hybrid Projection Decomposition (HPD). arXiv:2508.11935.
4. Yao et al. "Fully hardware-implemented memristor convolutional neural network." Nature 2020.
5. Wan et al. NeuRRAM: "A compute-in-memory chip based on resistive random-access memory." Nature 2022.
6. Rasch et al. "Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators." Nature Communications 2023.
7. Sun et al. "Solving matrix equations in one step with cross-point resistive arrays." PNAS 2019.
8. Sun et al. "One-step regression and classification with cross-point resistive memory arrays." Science Advances 2020.
9. Li et al. "Analog content-addressable memories with memristors." Nature Communications 2020.
10. Karunaratne et al. "In-memory hyperdimensional computing." Nature Electronics 2020.
11. Ielmini, Wong. "In-memory computing with resistive switching devices." Nature Electronics 2018.
12. Quamba. arXiv:2410.13229. Quamba2. arXiv:2503.22879.
13. Ma et al. BitNet b1.58: "The Era of 1-bit LLMs." 2024.
14. Pimba. arXiv:2507.10178. eMamba. arXiv:2508.10370. Loihi 2 S4D. arXiv:2409.15022.
15. Lin et al. (Zhongrui Wang group). RRAM zero-shot liquid state machine. Nature Computational Science 2025. DOI 10.1038/s43588-024-00751-z.

### 9.5 Sources in This Repository

1. `docs/ssm_cross_session_audit_20261003.md`
2. `docs/ssm_peft_round1_20261003.md`
3. `docs/ssm_debug_round2_20261003.md`
4. `docs/literature_audit_20261003.md`
5. `docs/SSM硬件算子与存内计算_前沿梳理.md`
6. `results/analysis/rram_quant_20261003/README.md`
