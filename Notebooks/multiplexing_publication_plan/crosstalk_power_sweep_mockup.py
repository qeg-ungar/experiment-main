"""
SCHEMATIC MOCKUP — illustrative synthetic data only, not experimental.

Illustrates the proposed power-sweep crosstalk-characterization figure:
  A) addressed-row Rabi oscillation at the current (low) drive power
  B) non-addressed-row ("crosstalk") oscillation at several drive powers,
     showing the low-power case sitting near the noise floor
  C) crosstalk amplitude vs Omega/delta, data vs the Omega^2/(Omega^2+delta^2) model
  D) effective (generalized) Rabi frequency vs Omega/delta, data vs sqrt(1+x^2) model
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---- fixed physical parameters (illustrative, matching the real dataset's scale) ----
delta_mhz = 1.4          # row-to-row gradient splitting
omega_list = [0.5, 1.0, 1.8]   # drive Rabi frequencies to compare (MHz); 0.5 = current low-power run
power_labels = ['0.5 MHz (current)', '1.0 MHz', '1.8 MHz']
power_colors = ['#9ECAE1', '#4292C6', '#08519C']   # sequential Blues ramp (ordered by power)
accent_color = '#D55E00'   # addressed-row / on-resonance series (Okabe-Ito vermillion)
model_color = '#888888'
noise_std = 0.0015
noise_floor = 3 * noise_std

decay_ns = 3000.0
t_vec = np.linspace(0, 6000, 60)


def damped_cosine(t, amp, freq_mhz, decay):
    return 1.0 + amp * np.cos(2 * np.pi * freq_mhz * 1e-3 * t) * np.exp(-t / decay)


def amp_fraction(x):
    # crosstalk amplitude relative to on-resonance amplitude
    return x**2 / (1 + x**2)


def freq_ratio(x):
    # Omega_eff / delta
    return np.sqrt(1 + x**2)


amp_addressed = 0.020  # on-resonance oscillation depth (a.u.)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ---------------------------------------------------------------------------
# Panel A: addressed row, on resonance, at the current operating power
# ---------------------------------------------------------------------------
ax = axes[0, 0]
omega0 = omega_list[0]
y = damped_cosine(t_vec, amp_addressed, omega0, decay_ns) + rng.normal(0, noise_std, t_vec.size)
ax.plot(t_vec, y, 'o-', ms=4, lw=1.2, color=accent_color, label=f'addressed row (Ω={omega0} MHz)')
ax.axhline(1.0, color=model_color, lw=0.8, ls=':')
ax.set_title('A. Addressed row — current operating power')
ax.set_xlabel('Pulse duration (ns)')
ax.set_ylabel('Contrast (a.u.)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower right')

# ---------------------------------------------------------------------------
# Panel B: non-addressed (neighboring) row at increasing drive power
# ---------------------------------------------------------------------------
ax = axes[0, 1]
for omega, color, label in zip(omega_list, power_colors, power_labels):
    f_eff = np.sqrt(omega**2 + delta_mhz**2)
    amp = amp_addressed * amp_fraction(omega / delta_mhz)
    y = damped_cosine(t_vec, amp, f_eff, decay_ns) + rng.normal(0, noise_std, t_vec.size)
    ax.plot(t_vec, y, 'o-', ms=4, lw=1.2, color=color, label=f'Ω={label}')

ax.axhspan(1 - noise_floor, 1 + noise_floor, color=model_color, alpha=0.15, lw=0,
           label='noise floor (±3σ)')
ax.annotate('barely above\nnoise floor', xy=(3900, 1.001), xytext=(700, 1.0115),
            fontsize=8, color=power_colors[0],
            arrowprops=dict(arrowstyle='->', color=power_colors[0], lw=1))
ax.set_title('B. Non-addressed row — crosstalk vs. drive power')
ax.set_xlabel('Pulse duration (ns)')
ax.set_ylabel('Contrast (a.u.)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)

# ---------------------------------------------------------------------------
# Panel C: crosstalk amplitude vs Omega/delta
# ---------------------------------------------------------------------------
ax = axes[1, 0]
x_curve = np.linspace(0, 1.6, 200)
ax.plot(x_curve, amp_fraction(x_curve), '-', lw=2, color=model_color,
         label=r'model: $\Omega^2/(\Omega^2+\delta^2)$')
ax.axhline(noise_floor / amp_addressed, color=model_color, lw=0.8, ls=':')
ax.text(1.55, noise_floor / amp_addressed + 0.02, 'noise floor', fontsize=8,
        color=model_color, ha='right')
ax.axvline(1.0, color=model_color, lw=0.8, ls='--', alpha=0.6)
ax.text(1.02, 0.05, r'$\Omega=\delta$', fontsize=8, color=model_color)

for omega, color, label in zip(omega_list, power_colors, power_labels):
    x = omega / delta_mhz
    y_true = amp_fraction(x)
    y_meas = y_true + rng.normal(0, 0.03)
    yerr = 0.04
    ax.errorbar(x, y_meas, yerr=yerr, fmt='o', ms=8, color=color, ecolor=color,
                capsize=3, label=f'Ω={label}')

ax.set_xlim(0, 1.6)
ax.set_ylim(-0.05, 1.05)
ax.set_title('C. Crosstalk amplitude — data vs. model')
ax.set_xlabel(r'$\Omega/\delta$')
ax.set_ylabel('Crosstalk amplitude / on-resonance amplitude')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')

# ---------------------------------------------------------------------------
# Panel D: effective Rabi frequency vs Omega/delta
# ---------------------------------------------------------------------------
ax = axes[1, 1]
ax.plot(x_curve, freq_ratio(x_curve), '-', lw=2, color=model_color,
         label=r'model: $\sqrt{1+(\Omega/\delta)^2}$')
ax.axvline(1.0, color=model_color, lw=0.8, ls='--', alpha=0.6)

for omega, color, label in zip(omega_list, power_colors, power_labels):
    x = omega / delta_mhz
    y_true = freq_ratio(x)
    y_meas = y_true + rng.normal(0, 0.03)
    yerr = 0.05
    ax.errorbar(x, y_meas, yerr=yerr, fmt='o', ms=8, color=color, ecolor=color,
                capsize=3, label=f'Ω={label}')

ax.set_xlim(0, 1.6)
ax.set_title('D. Effective (generalized) Rabi frequency — data vs. model')
ax.set_xlabel(r'$\Omega/\delta$')
ax.set_ylabel(r'$\Omega_{eff}/\delta$')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')

fig.suptitle('SCHEMATIC — illustrative synthetic data, not experimental\n'
             'Proposed power-sweep crosstalk characterization', fontsize=12, y=1.00)
fig.tight_layout()
out_path = Path(__file__).parent / 'crosstalk_power_sweep_mockup.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'saved to {out_path}')
