from Bessel import * 
import resonanzen as rs 
from roots_y import *
from tracing import *
import bound_states as bs
import numpy as np
import matplotlib.pyplot as plt
import tableReading
import WellenfunktionenPlots as WPlots
import os
import os.path


def heatmap_streuphase2(l, y, v0):
    '''Mathematische Implementierung des Sinusquadrats der Streuphase'''
    sigma = np.zeros_like(y) 
    a = rs.numerator(y, v0, l) / rs.denominator(y, v0, l)
    sin2 = 1 / (1 + 1/a**2)    
    # Wirkungsquerschnitt 
    sigma = ((4* np.pi)/y**2) *sin2
    
    return sin2, sigma






def plot_heatmap_streuphase(l, y_range, v0_range, num_y=100, num_v0=100, modus="y", save=True, plot_3d=False):
    '''Plot des Sinusquadrats der Streuphase (auch als 3D-Plot mit plot_3d=True) mit anzugebener y_range und v0_range als Zwei-Tupel mit Start und Endwert, der Auflösung in den Intervallen num_y und num_v0 sowie der Auftragung von v0 gegen x oder y durch modus.'''
    # Erzeuge Gitter
    y_vals = np.linspace(y_range[0], y_range[1], num_y)
    v0_vals = np.linspace(v0_range[0], v0_range[1], num_v0)
    YY, VV0 = np.meshgrid(y_vals, v0_vals)
    sin2_map = np.zeros_like(YY)

    # Berechnung sin^2 Gitterwerte
    for i in range(num_v0):
        for j in range(num_y):
            sin2_map[i, j], _ = heatmap_streuphase2(l, YY[i, j], VV0[i, j])

    # Transformation in (x,v0)-Raum
    XX = rs.to_x(YY, VV0)
    # Exponenzieren der Heatmap
    exp_sin2_map = np.exp(sin2_map + 1e-12)

    # 2D Heatmap Plot
    plt.figure(figsize=(10, 7))
    if modus == "x":
        im = plt.pcolormesh(XX, VV0, exp_sin2_map, shading='auto', cmap='viridis')
        plt.xlabel('x')
    else:
        im = plt.pcolormesh(YY, VV0, exp_sin2_map, shading='auto', cmap='viridis')
        plt.xlabel('$y_R$')
    plt.ylabel('$v_0$')
    plt.colorbar(im, label=r'$e^{\sin^2 \delta}$')
    plt.title(f'Exponenzierte Heatmap von $\sin^2 \delta_l$ für l={l}')
    plt.legend(loc='upper right')
    plt.xlim(0, y_range[1])
    plt.tight_layout()

    # Lade Resonanzdaten
    x_rs, y_rs, v0_rs = tableReading.read_order_Resonanzen(l)
    # Maskiere relevante Wertebereiche
    mask_rs = (x_rs >= XX.min()) & (x_rs <= XX.max()) & (v0_rs >= v0_range[0]) & (v0_rs <= v0_range[1])
    x_plot_rs = x_rs[mask_rs]
    y_plot_rs = y_rs[mask_rs]
    v0_plot_rs = v0_rs[mask_rs]

    if modus == "x":
        plt.scatter(x_plot_rs, v0_plot_rs, c='red', s=1, label='Resonances')
    else:
        plt.scatter(y_plot_rs, v0_plot_rs, c='red', s=1, label='Resonances')

    # Speicher 2D Heatmap
    if save:
        save_dir = os.path.join(os.getcwd(), "Plots_Heatmap")
        os.makedirs(save_dir, exist_ok=True)
        filename_base = f"Heatmap_sin2_{modus}_l{l}"
        plt.savefig(os.path.join(save_dir, filename_base + ".png"), dpi=200)

    # Wenn angegeben: Erzeugung 3D-Plot
    if plot_3d:
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
    
        # Oberfläche mit leichter Transparenz
        surf = ax.plot_surface(
            YY, VV0, sin2_map,
            cmap='viridis',
            rcount=100, ccount=100,
            alpha=0.6,
            linewidth=0,
            antialiased=True
        )
        ax.set_xlabel('y')
        ax.set_ylabel('$v_0$')
        ax.set_zlabel(r'$\sin^2\delta_l$')
        ax.set_title(f'3D-Plot von $\sin^2 \delta_l$ für l={l}')
    
        # Iteration über alle v0-Werte als direkte Ergängzung zu Abschnitt in Bachelorarbeit über 2D Plot der Streuphase gegen y (hier ebenfalls eingetragene Konturen), (nur für l=4)
        if l == 4:
            v0_list = [22.94, 30, 48]
            colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for v0_fixed, color in zip(v0_list, colors):
                v0_vals = VV0[:, 0]
                idx     = np.argmin(np.abs(v0_vals - v0_fixed))

                y_line  = YY[idx, :]
                v0_line = VV0[idx, :]
                z_line  = sin2_map[idx, :]

                ax.plot(
                    y_line, v0_line, z_line,
                    linewidth=2,
                    color=color,
                    label=fr'$v_0 = {v0_fixed}$'
                )

        ax.legend(loc='upper left', frameon=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
        # Speicherung 3D-Plot
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_dir, filename_base + "_3D_multi.png"), dpi=200)


# Aufrufe zur Reproduktion der Heatmaps und des 3D-Plots in der Bachelorarbeit:
l=0
while l < 11:
    plot_heatmap_streuphase(l, y_range=(0.01, 30), v0_range=(0, 500), modus="y", plot_3d= True)
    l += 1
plot_heatmap_streuphase(l=4, y_range=(0.00001, 30), num_y=10000, v0_range=(0, 500), modus="y", plot_3d= True)