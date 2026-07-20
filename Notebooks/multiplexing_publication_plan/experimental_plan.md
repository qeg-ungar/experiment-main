# NV-array multiplexing — experimental plan for publication

## Context

Goal: demonstrate frequency-multiplexed addressing of individual rows of NV
centers imaged on a SPAD array, using a static magnetic field gradient
(currently 1-dimensional, so each row of the array has a distinct ODMR/Rabi
drive frequency).

Existing data (three `SPAD_time_rabi_npzprocess_batch.ipynb` runs, each a
single drive frequency tuned to one row, 7-10 tracking iterations each):

| Drive frequency targets | Files (`gradient_tracking` tag) |
|---|---|
| Row 1 (bottom row) | `486_time_rabi_gradient_tracking_191405` |
| Row 2 (middle row) | `493_time_rabi_gradient_tracking_230359` |
| Row 3 (top row) | `503_time_rabi_gradient_tracking_085223` |

Measured parameters: row-to-row gradient splitting **δ ≈ 1.4 MHz**,
drive Rabi frequency **Ω ≈ 0.5 MHz** (chosen as the smallest power that still
gives a clean on-resonance Rabi oscillation, in order to minimize crosstalk
for the addressing demonstration).

For each dataset, a 3×3 pixel grid centered on the array's brightest pixel is
fit with a damped cosine; the FFT of the contrast trace shows a clear peak at
Ω for the addressed row and a small peak (consistent with off-resonant
`Ω²/(Ω²+δ²)` driving) for the non-addressed rows.

## Priority-ordered plan

1. **Crosstalk matrix (reprocess existing data — no new acquisition needed)**
   Build a 3×3 grid: drive frequency (row) × readout row (column). The
   diagonal is the existing on-resonance fits; the off-diagonal cells are
   the same three datasets' non-addressed-row contrast, fit for amplitude and
   effective Rabi frequency `Ω_eff = √(Ω²+δ²)`. **All nine cells are reported
   individually — they are not averaged together.** That said, the off-diagonal
   cells naturally group by nominal δ: four instances share δ≈1.4 MHz
   (nearest-neighbor — driving row 1 or row 3 gives one neighbor each, driving
   row 2 gives two) and two instances share δ≈2.8 MHz (next-nearest-neighbor,
   the two corner cells). Because the isotropic single-δ, single-Ω model
   predicts the same crosstalk amplitude/frequency for all cells within a
   group, the spread *within* each group is a useful internal consistency
   check — if the four nearest-neighbor cells disagree with each other by
   more than their individual fit uncertainties, that points to something the
   simple model doesn't capture (uneven drive power across rows, a
   miscalibrated or nonlinear gradient, etc.) and is worth chasing down before
   trusting the model. This matrix is the central publication figure for the
   *selectivity* claim.

   **Fitting note (low-SNR off-diagonal cells):** at the current low drive
   power the crosstalk peak can be only marginally above the noise floor, and
   letting a nonlinear fit (5 free parameters, including frequency) run on a
   near-noise trace risks locking onto a spurious frequency and reporting a
   confident but meaningless amplitude. Better: fix the frequency at the
   theoretically predicted `Ω_eff = √(Ω²+δ²)` (using Ω from the high-SNR
   on-resonance fit and δ from the gradient calibration) and fit only
   amplitude + phase via *linear* least squares
   (`y = offset + a·cos(2π·Ω_eff·t) + b·sin(2π·Ω_eff·t)`, solved with
   `np.linalg.lstsq`). This is a matched-filter/lock-in-style estimator and
   gives a trustworthy covariance-based error bar even when SNR ~ 1. If the
   resulting amplitude is not several σ above zero, report it as an upper
   limit rather than a point estimate.

2. **Simultaneous multi-tone driving (new acquisition — the actual
   multiplexing demo)**
   Everything so far is sequential single-tone addressing. Apply a
   multi-tone drive (sum of 2-3 frequencies) in a single shot and show
   independent control — e.g. row 1 driven to a π-pulse while row 3 is
   driven to a different target angle simultaneously, read out together.
   This is what distinguishes "spectrally distinguishable" from "multiplexed
   control."

3. **Full-frame difference imaging**
   Apply the multi-tone pulse combination and take a full-sensor
   (32×32) difference image (signal − reference) instead of only the 3×3
   crop, to visually show that the contrast change is spatially localized to
   the addressed row(s).

4. **Optical vs. spin crosstalk control**
   With the drive off, measure the point-spread-function bleed-through
   between rows (pure imaging crosstalk) so the residual can be
   attributed purely to `Ω²/(Ω²+δ²)` spin crosstalk rather than partly to
   optical crosstalk from the imaging system.

5. **Power-sweep crosstalk characterization (new acquisition)**
   Fix δ (nearest-neighbor row pair) and sweep drive amplitude to vary Ω
   across several settings, from the current low-power point up toward
   Ω≈δ. Fit crosstalk amplitude and Ω_eff at each power and overlay against
   the analytic model curves — see "Figure layout" below. This both
   validates the model over a wider range than a single point and gives a
   practical selectivity-vs-gate-speed design curve. Complementary to item 1:
   item 1 sweeps δ at fixed Ω, this sweeps Ω at fixed δ.

6. **Error bars / statistics**
   Each dataset already has 7-10 independent tracking iterations, currently
   collapsed into one weighted-average contrast curve. Extract per-iteration
   contrast curves and propagate iteration-to-iteration scatter into fit
   uncertainties on Ω, δ, and π-pulse time for every figure above.

7. **(Stretch) scale beyond 3 rows**
   If the gradient range and camera field of view allow more than three
   spectrally distinct rows, demonstrating N > 3 addressable rows
   substantially strengthens the scalability claim.

Suggested order: (1) and (6) first (pure reprocessing of existing data),
then (4) as a quick control, then (2) and (3) as the new "hero" experiment,
then (5) and (7) as time permits.

## Figure layout

### Figure 1 — crosstalk matrix (item 1, real data)
3×3 grid or heatmap, rows = drive frequency, columns = readout row.
Each cell: contrast trace + fit, with fitted amplitude and Ω_eff annotated.
Diagonal = on-resonance (selectivity demo); off-diagonal = crosstalk,
grouped by δ (nearest- vs. next-nearest-neighbor).

### Figure 2 — power-sweep crosstalk characterization (item 5, new data)
Four-panel figure (mockup below shows the intended layout with synthetic
data — see `crosstalk_power_sweep_mockup.png`):

- **A.** Addressed-row contrast vs. pulse duration at the current operating
  power (on-resonance Rabi oscillation, single curve).
- **B.** Non-addressed (nearest-neighbor) row contrast vs. pulse duration at
  several drive powers, sequential color ramp low→high power. Shows the
  low-power crosstalk oscillation sitting near the noise floor, growing
  clearly visible at higher power.
- **C.** Crosstalk amplitude (normalized to on-resonance amplitude) vs.
  Ω/δ — analytic curve `Ω²/(Ω²+δ²)` overlaid with data points per power
  setting. At the current power, plot all four nearest-neighbor pairs
  (item 1) as separate markers clustered at that Ω/δ rather than averaging
  them into one point — their spread is the empirical error bar and
  doubles as the same model consistency check described in item 1.
- **D.** Effective Rabi frequency Ω_eff/δ vs. Ω/δ — analytic curve
  `√(1+(Ω/δ)²)` overlaid with the same data points (same four-marker
  treatment at the current power). Frequency is generally an easier/more
  robust fit parameter than amplitude at low power, since it comes from the
  oscillation period rather than signal size.

### Figure 3 — multi-tone multiplexed control (item 2, new data)
Simultaneous multi-frequency drive; per-row contrast readout on one shot
showing independent rotation angles per row.

### Figure 4 — full-frame difference image (item 3, new data)
Full 32×32 sensor difference image before/after the multi-tone pulse,
showing spatial localization of the contrast change to the addressed row(s).

## Files in this directory

- `experimental_plan.md` — this file
- `crosstalk_power_sweep_mockup.py` — script generating the illustrative
  (synthetic-data) mockup of Figure 2
- `crosstalk_power_sweep_mockup.png` — rendered mockup of Figure 2
