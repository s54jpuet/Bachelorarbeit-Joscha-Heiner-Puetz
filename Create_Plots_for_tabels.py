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
    Außerdem entweder bs oder rs mittels typ sowie beide zusammen mit mode = 'both'
    oder einzeln mit mode = 'single'.
    """

    # 1) Wahl der Abszisse und Beschriftung
    if axis == 'y':
        col_idx = 1
        xlabel = r'$y$'
    else:
        col_idx = 0
        if mode == 'single':
            if typ == "Resonanzen":
                xlabel = r'$x_R = \sqrt{y_R^2 + v_0}$ für Resonanzen'
            elif typ == "Bound States":
                xlabel = r'$x_B = \sqrt{v_0 - y_B^2}$ für gebundene Zustände'
            elif typ == "Bs_mit_trace_root":  # temporär
                xlabel = r'$x_B = \sqrt{v_0 - y_B^2}$ für gebundene Zustände'
        else:
            xlabel = (r'$x_R = \sqrt{y_R^2 + v_0}$ für Resonanzen,  '
                      r'$x_B = \sqrt{v_0 - y_B^2}$ für gebundene Zustände')
    
    # 2) Farben und Figuren-Setup
    # Für 'single' bleibt die bisherige Farbwahl (nach l).
    # Für 'both' bekommen Resonanzen und gebundene Zustände feste, verschiedene Farben.
    max_colors = 10  
    colormap = plt.cm.tab10  
    colorlist = [colormap(k / max_colors) for k in range(max_colors)]

    # feste Typ-Farben für gemeinsamen Plot
    type_colors = {
        "Resonanzen": "#1f77b4",        # tab:blue
        "Bound States": "#ff7f0e",      # tab:orange
        "Bs_mit_trace_root": "#2ca02c"  # tab:green (falls genutzt)
    }

    plt.figure(figsize=(12, 8))
    plt.xlabel(xlabel)
    plt.ylabel(r'$v_0$')

    # 3) Dateien einlesen und plotten
    i = 0
    # Flags, um doppelte Legendeneinträge zu vermeiden
    rs_labeled = False
    bs_labeled = False
    bs2_labeled = False

    while os.path.isfile(tableReading.f_name(typ, l, i)):
        # Hier liest read_line_Resonanzen immer alle drei Spalten ein:
        x_rs, y_rs, v0_rs = tableReading.read_line_Resonanzen(l, i)
        x_bs, y_bs, v0_bs = tableReading.read_line_Bound_States(l, i)
        x_bs2, y_bs2, v0_bs2 = tableReading.read_line_Bound_States2(l, i)  # temporär

        # je nach axis wählen wir entweder x oder y
        abscissa_rs = x_rs if col_idx == 0 else y_rs
        abscissa_bs = x_bs if col_idx == 0 else y_bs
        abscissa_bs2 = x_bs2 if col_idx == 0 else y_bs2  # temporär
        
        # Farben
        color_l = colorlist[l % max_colors]  # für 'single'
        color_rs = type_colors["Resonanzen"]
        color_bs = type_colors["Bound States"]
        color_bs2 = type_colors["Bs_mit_trace_root"]

        min_v0 = np.min(v0_rs)

        if mode == 'single' and typ == 'Resonanzen':
            label = 'Resonanzen' if not rs_labeled else None
            plt.plot(abscissa_rs, v0_rs, '-', markersize=1, color=color_l, label=label)
            rs_labeled = True
            # Horizontale Linie bei Minimum v0 (ohne Label, damit die Legende schlank bleibt)
            plt.axhline(min_v0, color=color_l, linestyle='--', linewidth=1)

        elif mode == 'single' and typ == 'Bound States':
            label = 'gebundene Zustände' if not bs_labeled else None
            plt.plot(abscissa_bs, v0_bs, 'o', markersize=1, color=color_l, label=label)
            bs_labeled = True

        elif mode == 'single' and typ == 'Bs_mit_trace_root':  # temporär
            label = 'gebundene Zustände (trace root)' if not bs2_labeled else None
            plt.plot(abscissa_bs2, v0_bs2, '-', markersize=1, color=color_l, label=label)
            bs2_labeled = True

        elif mode == 'both':
            # Unterschiedliche, feste Farben für die Typen + einmalige Labels
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
    # Legend nur zeigen, wenn auch Labels vorhanden sind
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.xlim(0, 50)
    plt.ylim(0, 1000)

    # 4) Speichern
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


# Beispielaufrufe (wie bei dir):
j=0
while j < 11:
    #plot_for_tables_of_xy("Bs_mit_trace_root", j, "x", "single")
    #plot_for_tables_of_xy("Bs_mit_trace_root", j, "y", "single")
    #plot_for_tables_of_xy("Resonanzen", j, "x")
    #plot_for_tables_of_xy("Resonanzen", j, "y")
    #plot_for_tables_of_xy("Bound States", j, "x")
    #plot_for_tables_of_xy("Bound States", j, "y")
    plot_for_tables_of_xy("Resonanzen", j, "x", "both")
    plot_for_tables_of_xy("Resonanzen", j, "y", "both")
    j += 1

#plot_for_tables_of_xy("Resonanzen", 4, "x", "both")
#plot_for_tables_of_xy("Resonanzen", 0, "x", "both")
#plot_for_tables_of_xy("Resonanzen", 4, "y", "both")
#plot_for_tables_of_xy("Resonanzen", 0, "y", "both")
