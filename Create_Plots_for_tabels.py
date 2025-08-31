from Bessel import * 
import resonanzen as rs 
from roots_y import *
from tracing import *
import bound_states as bs
import numpy as np
import matplotlib.pyplot as plt
import os
import tableReading



def plot_for_tables_of_xy(typ, l, axis='x', mode='single', save=True):
    """
    Plottet v0 gegen entweder x (erste Spalte) oder y (zweite Spalte).
    Außerdem entweder gebundene Zustände (typ = "Bound States") oder Resonanzen (typ = "Resonanzen") mittels typ sowie beide zusammen mit mode = 'both'
    oder einzeln mit mode = 'single'.
    """

    # Wahl der Abszisse und Achsenbeschriftung
    if axis == 'y':
        col_idx = 1
        xlabel = r'$y_{R/B}$'
    else:
        col_idx = 0
        if mode == 'single':
            if typ == "Resonanzen":
                xlabel = r'$x_R = \sqrt{y_R^2 + v_0}$ für Resonanzen'
            elif typ == "Bound States":
                xlabel = r'$x_B = \sqrt{v_0 - y_B^2}$ für gebundene Zustände'
        else:
            xlabel = r'$x_{R/B}$'
    
    # Farbfestlegung
    max_colors = 10  
    colormap = plt.cm.tab10  
    colorlist = [colormap(k / max_colors) for k in range(max_colors)]

    # feste Typfarben für gemeinsamen Plot
    typ_colors = {
        "Resonanzen": "#1f77b4",        # tab:blue
        "Bound States": "#ff7f0e",      # tab:orange
    }

    plt.figure(figsize=(12, 8))
    plt.xlabel(xlabel)
    plt.ylabel(r'$v_0$')

    # Daten einlesen und plotten
    i = 0
    # Flags zur Vermeidung doppelter Legendeneinträge
    rs_labeled = False
    bs_labeled = False
    bs2_labeled = False

    while os.path.isfile(tableReading.f_name(typ, l, i)):
        x_rs, y_rs, v0_rs = tableReading.read_line_Resonanzen(l, i)
        x_bs, y_bs, v0_bs = tableReading.read_line_Bound_States(l, i)

        # je nach angegebener Achse wird x oder y verwendet
        abscissa_rs = x_rs if col_idx == 0 else y_rs
        abscissa_bs = x_bs if col_idx == 0 else y_bs
        
        color_l = colorlist[l % max_colors]
        color_rs = typ_colors["Resonanzen"]
        color_bs = typ_colors["Bound States"]

        min_v0 = np.min(v0_rs)

        if mode == 'single' and typ == 'Resonanzen':
            label = 'Resonanzen' if not rs_labeled else None
            plt.plot(abscissa_rs, v0_rs, '-', markersize=1, color=color_l, label=label)
            rs_labeled = True
            # Horizontale Linie bei Minimum v0
            plt.axhline(min_v0, color=color_l, linestyle='--', linewidth=1)
        elif mode == 'single' and typ == 'Bound States':
            label = 'gebundene Zustände' if not bs_labeled else None
            plt.plot(abscissa_bs, v0_bs, 'o', markersize=1, color=color_l, label=label)
            bs_labeled = True
        elif mode == 'both':
            label_bs = 'gebundene Zustände' if not bs_labeled else None
            label_rs = 'Resonanzen' if not rs_labeled else None

            plt.plot(abscissa_bs, v0_bs, 'o', markersize=1, color=color_bs, label=label_bs)
            plt.plot(abscissa_rs, v0_rs, '-', markersize=1, color=color_rs, label=label_rs)
            # Min-Linie ohne Label
            plt.axhline(min_v0, color=color_rs, linestyle='--', linewidth=1)

            bs_labeled = True
            rs_labeled = True

        i += 1

    plt.title(f"Konturplot der Ordnung l={l} gegen {axis}")
    # Legend wird nur gezeigt, wenn Labels vorhanden sind
    handles, _labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Festlegung Plotbereich
    plt.xlim(0, 50)
    plt.ylim(0, 1000)

    # Speichern
    if save:
        save_dir = os.path.join(os.getcwd(), "Plots_zu_Tabellen")
        os.makedirs(save_dir, exist_ok=True)
        if mode == 'single':
            filename = os.path.join(
                save_dir,
                f"Plot_{typ}_fuer_l={l}_gegen_{axis}.png"
            )
        elif mode == "both":
            filename = os.path.join(
                save_dir,
                f"Plot_alles_fuer_l={l}_gegen_{axis}.png"
            )
        plt.savefig(filename, dpi=200)
        plt.close()


# Beispielaufrufe zur Erzeugung der separaten sowie gemischten Konturplots aus der Bachelorarbeit:
j=0
while j < 11:
    plot_for_tables_of_xy("Resonanzen", j, "x")
    plot_for_tables_of_xy("Resonanzen", j, "y")
    plot_for_tables_of_xy("Bound States", j, "x")
    plot_for_tables_of_xy("Bound States", j, "y")
    plot_for_tables_of_xy("Resonanzen", j, "x", "both")
    plot_for_tables_of_xy("Resonanzen", j, "y", "both")
    j += 1