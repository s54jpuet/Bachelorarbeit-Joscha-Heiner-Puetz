from Bessel import * 
import resonanzen as rs 
from roots_y import *
from tracing import *
import bound_states as bs
import numpy as np
import matplotlib.pyplot as plt
import tableReading
import os

def rs_plot_wavefunction(l_list, v0, x_max_intervall):
 	
    '''
    Diese Funktion plottet die aus den berechneten Konturplots der Resonanzen folgenden Wellenfunktionen aller Lösungen für ein spezifisches v0
    '''
	
    all_a = {}  # korrespondierend zu x=qr0
    all_b = {}  # korrespondierend zu y=k*r0
    for i in l_list:
        x, y = tableReading.find_xy_for_v0("Resonanzen", i, v0)
        all_a[i] = x
        all_b[i] = y
 
 
    # Erzeugung des Plots
    x_plot = np.linspace(0, x_max_intervall, x_max_intervall*100)  # gewünschtes Intervall
    for l in all_a:
         for a, b in zip(all_a[l], all_b[l]):
             y_plot = rs.radial_of_resonance(l, x_plot, a, b)
             plt.plot(x_plot, y_plot, label=fr"$l={l},\, x = q r_0={a:.2f},\, y = k r_0={b:.2f}$")
 
    plt.title(f"Radiale Wellenfunktion der Resonanzen f\u00fcr $v_0={v0}$")
    plt.xlabel(r'$\frac{r}{r_0}$')
    plt.ylabel(r'$\frac{r}{r_0} \cdot R_l$')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
 
    # Speichern des Plots
    out_dir = os.path.join("Plots_Wellenfunktionen")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"Resonanzen_v0_{v0}.png"
    path = os.path.join(out_dir, filename)   
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {path}")
 
def bs_plot_wavefunction(l_list, v0, x_max_intervall):
    '''
    Diese Funktion Plottet die aus den berechneten Konturplots der Bound States folgenden Wellenfunktionen aller Lösungen für ein spezifisches v0
    '''

    all_a = {}  # korrespondierend zu x=qr0
    all_b = {}  # korrespondierend zu y=\kappa*r0
    for i in l_list:
        x, y = tableReading.find_xy_for_v0("Bound States", i, v0)

        all_a[i] = x
        all_b[i] = y

    # Erzeugung des Plots
    x = np.linspace(0, x_max_intervall, x_max_intervall*200)  # gewünschtes Intervall
    for l in all_a:
        for a, b in zip(all_a[l], all_b[l]):
            y = bs.radial_of_bound_states(l, x, a, b)   
            plt.plot(x, y, label=fr"$l={l},\, x = q r_0={a:.2f},\, y = \kappa r_0={b:.2f}$")

    plt.title(f"Radiale Wellenfunktion der gebundenen Zust\u00e4nde f\u00fcr $v_0={v0}$")
    plt.xlabel(r'$\frac{r}{r_0}$')
    plt.ylabel(r'$\frac{r}{r_0} \cdot R_l$')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Speichern des Plots
    out_dir = os.path.join("Plots_Wellenfunktionen")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"boundstates_v0_{v0}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {path}")
