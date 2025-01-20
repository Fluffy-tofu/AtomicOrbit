import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar, m_e

def calculate_magnetic_terms(B_values, m_l=1, r=5.29177e-11):
    mu_B = 5.788e-5  # Bohrsches Magneton in eV/T
    para_term = mu_B * B_values * m_l
    dia_term = (e**2 * B_values**2 * r**2) / (8 * m_e)
    return para_term, dia_term

# Erstelle zwei separate Plots
B_values = np.linspace(0, 100, 1000)
para_term, dia_term = calculate_magnetic_terms(B_values)

# Paramagnetischer Term
plt.figure(figsize=(8, 6))
plt.plot(B_values, para_term, 'b-', linewidth=1.5)
plt.xlabel('Magnetfeldstärke (T)')
plt.ylabel('Energie (eV)')
plt.title('Paramagnetischer Term (∝ B)')
plt.grid(True, alpha=0.2, linestyle='--')
plt.xlim(0, 100)
plt.ylim(0, max(para_term)*1.05)
plt.tight_layout()
plt.savefig('paramagnetic_term.png', dpi=300, bbox_inches='tight')
plt.close()

# Diamagnetischer Term
plt.figure(figsize=(8, 6))
plt.plot(B_values, dia_term, 'r-', linewidth=1.5)
plt.xlabel('Magnetfeldstärke (T)')
plt.ylabel('Energie (eV)')
plt.title('Diamagnetischer Term (∝ B²)')
plt.grid(True, alpha=0.2, linestyle='--')
plt.xlim(0, 100)
plt.ylim(0, max(dia_term)*1.05)
plt.tight_layout()
plt.savefig('diamagnetic_term.png', dpi=300, bbox_inches='tight')
plt.close()