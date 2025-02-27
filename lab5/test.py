import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple

# Set style for scientific publication plots
set_science_style(use_tex=False)  # Disable tex for now since we're in notebook

# Calculate temperatures from Wien's law
b_nm = 2.898e-3 * 1e9  # Wien's constant in nm·K
T = b_nm / peak_wavelength  # Temperature in Kelvin
T4 = T**4  # T^4 for Stefan-Boltzmann

# Calculate electrical power
P_electrical = voltage * current  # Power in Watts

# Normalize T4 values to prevent numerical issues
T4_scale = 1e14  # Scale factor based on the order of magnitude of T4
T4_normalized = T4 / T4_scale

# Create figure with publication-friendly dimensions
fig, ax = plt.subplots(figsize=get_figure_size('single', ratio=1.2))

# Plot data with high contrast colors and appropriate marker size
ax.scatter(P_electrical, T4,   # Note: Still plotting original T4 values
          c=HIGH_CONTRAST_COLORS[0],
          s=MARKER_SIZES['normal']**2,
          alpha=0.8,
          label='Measured Data')

# Fit linear relationship with normalized data
pfit, pcov = curve_fit(lambda x, m, b: m*x + b, P_electrical, T4_normalized)
perr = np.sqrt(np.diag(pcov))

# Create fit line (scale back up for plotting)
x_fit = np.linspace(min(P_electrical), max(P_electrical), 100)
y_fit = (pfit[0]*x_fit + pfit[1]) * T4_scale

# Plot fit line
ax.plot(x_fit, y_fit, '--', 
        color=HIGH_CONTRAST_COLORS[1],
        linewidth=LINE_WIDTHS['normal'],
        alpha=0.8,
        label=f'Linear Fit')

# Customize plot
ax.set_xlabel('Electrical Power (W)')
ax.set_ylabel('Temperature⁴ (K⁴)')
ax.set_title('Stefan-Boltzmann Relationship in Halogen Lamp')
ax.grid(True, alpha=0.3)
ax.legend()

# Add fit parameters as text box (scale back up for display)
fit_text = f'T⁴ = ({pfit[0]*T4_scale:.2e} ± {perr[0]*T4_scale:.1e})P + ({pfit[1]*T4_scale:.2e} ± {perr[1]*T4_scale:.1e})'
ax.text(0.05, 0.95, fit_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print analysis
print(f"Temperature range: {min(T):.0f}K to {max(T):.0f}K")
print(f"Power range: {min(P_electrical):.2f}W to {max(P_electrical):.2f}W")