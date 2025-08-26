import os
import os.path
import numpy as np
import pandas as pd
import tableReading as tR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def first_v0s(l):
    i = 0
    first_v0s = []
    while os.path.isfile(tR.f_name("Bound States", l, i)):
        try:
            filename = tR.f_name("Bound States", l, i)
            data = np.loadtxt(filename, delimiter=",", skiprows=1)
            data = np.transpose(data)
            data2 = data[2]
            first_v0s.append(data2[0])

        except Exception as e:
            print(f"Fehler beim Lesen der Datei Typ: Resonanzen mit l={l} und i={i}")
            raise e
        i = i+1
    #print(first_v0s)                                                        # gibt alle v0 eines l bei neuen rs aus
    return first_v0s

#first_v0s(0)



def division_of_next_bound_state(l):
    '''
    Numerischer Beweis, dass asymptotisch abstände zwischen Gliedern bzgl v0 1 approachen
    '''
    v0s = np.array(first_v0s(l))
    indices = np.arange(len(v0s) - 1)
    divided_v0s = np.empty(len(v0s)-1)
    for i in range(len(v0s)-1):
        divided_v0s[i] = v0s[i+1]/v0s[i]

    plt.figure()
    plt.plot(indices, divided_v0s, marker='o')
    plt.xlabel('Index')
    plt.ylabel('v0[n+1] / v0[n]')
    plt.title('Verhältnis aufeinanderfolgender v0s')
    plt.grid(True)
    plt.show()
    return divided_v0s

#division_of_next_bound_state(0)


def eval_c_of_forecast_for_all_l(typ):
    # ... dein bisheriger Code, der csv_path setzt und df lädt ...
    # Beispiel (anpassen falls bei dir anders):
    save_dir = os.path.join(os.getcwd(), "Prognosefunktionen")
    safe_typ = typ.replace(" ", "")
    safe_kind = "limk0"  # oder was du oben verwendest
    csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.csv")
    df = pd.read_csv(csv_path, index_col='l')

    # --- NEU: sauber extrahieren & NaNs entfernen ---
    # Wir brauchen l (Index) und c = c_quad als float, ohne NaNs/Inf
    l_vals = pd.to_numeric(df.index, errors='coerce').to_numpy(dtype=float)
    c = pd.to_numeric(df.get('c_quad', pd.Series(index=df.index, dtype=float)), errors='coerce').to_numpy(dtype=float)

    mask = np.isfinite(l_vals) & np.isfinite(c)
    l_vals = l_vals[mask]
    c = c[mask]

    # Falls zu wenige Punkte vorhanden sind, defensiv zurückgeben
    if l_vals.size < 2:
        # zu wenig für sinnvolle Regression
        return 0.0, 0.0, 0.0, float(np.nan)

    # Grad des Fits defensiv wählen (max 3, aber < Anzahl Punkte)
    deg = min(3, int(l_vals.size) - 1)
    coef = np.polyfit(l_vals, c, deg=deg)  # höchster Grad zuerst

    # auf kubische Koeffizienten (a3 l^3 + b3 l^2 + n3 l + m3) abbilden
    # fehlende Grade mit 0 auffüllen
    coef_full = np.zeros(4, dtype=float)
    coef_full[-(deg+1):] = coef  # rechtsbündig einsetzen
    a3, b3, n3, m3 = coef_full  # genau 4 Werte

    # R^2 nur berechnen, wenn mind. 2 Punkte & kein NaN
    try:
        c_fit = np.polyval(coef_full, l_vals)
        from sklearn.metrics import r2_score
        r2_3 = r2_score(c, c_fit) if l_vals.size >= 2 else float('nan')
    except Exception:
        r2_3 = float('nan')

    # Wenn du r2 irgendwo ausgibst, gib r2_3 zurück; sonst nur Koeffizienten:
    return a3, b3, n3, m3

#eval_c_of_forecast_for_all_l("Resonanzen")



# ---- Helfer ----
def _ensure_columns(df, cols, index_name='l'):
    """Sorgt dafür, dass df exakt die Spalten 'cols' (in dieser Reihenfolge) hat."""
    if df is None:
        out = pd.DataFrame(columns=cols)
        out.index.name = index_name
        return out
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df.index.name = index_name
    return df

def safe_fmt_6(x):
    """Formatiert Zahlen auf 6 Nachkommastellen; Strings (z. B. Formeln) bleiben unverändert."""
    try:
        return f"{float(x):.6f}"
    except (ValueError, TypeError):
        return str(x) if pd.notna(x) else ""

# ---- Hauptfunktion ----
def v0_forecast_func(typ, l, kind="lim_k_0"):
    """
    Quadratischer Fit ohne Fehler.
    - Plot: Datenpunkte + Fit (keine Errorbars, keine Sondermarkierungen)
    - CSV (Haupt): a_quad, b_quad, c_quad, v0_first (numerisch)
    - LaTeX (rounded): a,b als Vielfache von pi^2; c, v0_first mit 6 Nachkommastellen
    """
    # 1) Datei & Titel
    ordner = f"tables/l={l}"
    if kind == "lim_k_0":
        infile = os.path.join(ordner, f"neue_{typ}_l={l}_mit_Fehler.csv")
        plot_title = f"Prognosefunktion lim_k_0 {typ} l={l}"
    elif kind == "min_v0":
        infile = os.path.join(ordner, f"min_v0_{typ}_l={l}.csv")
        plot_title = f"Prognosefunktion min_v0 {typ} l={l}"
    else:
        raise ValueError(f"Unbekanntes kind='{kind}'")

    # 2) Daten einlesen
    data = np.loadtxt(infile, delimiter=",", skiprows=1)
    # erwartetes Layout: lim_k_0 -> [x, y, v0_first, (sigma_v0)], min_v0 -> [x, y, v0_first]
    v0s = data[:200, 2].astype(float)
    if v0s.size == 0:
        raise ValueError(f"Keine v0-Daten in {infile} gefunden.")
    indices2 = np.arange(1, len(v0s) + 1, dtype=float)

    # 3) Quadratischer Fit (ungewichtet)
    coeffs2 = np.polyfit(indices2, v0s, deg=2)  # [a,b,c]
    y_fit = np.polyval(coeffs2, indices2)

    # 4) Werte für Tabellen
    v0_first = float(v0s[0])
    c_value  = float(coeffs2[2])

    # 5) a,b als Vielfache von π² (nur für „rounded“-Tabellen)
    pi2 = np.pi**2
    latex_ab = {
        'a_quad': rf"${int(np.round(coeffs2[0] / pi2))}\pi^2$",
        'b_quad': rf"${int(np.round(coeffs2[1] / pi2))}\pi^2$"
    }

    # 6) Plot (ohne Sondermarkierung)
    plt.figure(figsize=(7,5))
    plt.plot(indices2, v0s, 'x', markersize=3, label='$v_0$ Daten')
    plt.plot(indices2, y_fit, '--',
             label=(f'Quad: a={coeffs2[0]:.6g}, '
                    f'b={coeffs2[1]:.6g}, '
                    f'c={coeffs2[2]:.6g}'))
    plt.xlabel(f"# neue {typ}")
    plt.ylabel("$v_0$")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_dir = os.path.join(os.getcwd(), "Prognosefunktionen")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, plot_title.replace(" ", "_") + ".png"), dpi=200)
    plt.close()

    # 7) Haupttabelle (numerisch identisch für beide Typen)
    safe_typ  = typ.replace(" ", "")
    safe_kind = kind.replace("_","")
    csv_path  = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.csv")
    tex_path  = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.tex")

    main_cols = ['a_quad', 'b_quad', 'c_quad', 'v0_first']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='l')
    else:
        df = None
    df = _ensure_columns(df, main_cols, index_name='l')

    try:
        l_key = int(l)
    except Exception:
        l_key = l
    df.loc[l_key] = [float(coeffs2[0]), float(coeffs2[1]), c_value, v0_first]

    # sortiere bei ganzzahligem Index
    try:
        df.index = df.index.astype(int)
        df.sort_index(inplace=True)
    except Exception:
        pass
    df.to_csv(csv_path)

    latex_headers = {
        'a_quad': r'$a_{\mathrm{quad}}$',
        'b_quad': r'$b_{\mathrm{quad}}$',
        'c_quad': r'$c_{\mathrm{quad}}$',
        'v0_first': r'$v_0^{(1)}$'
    }
    df_latex = df.rename(columns=latex_headers)
    with open(tex_path, 'w') as f:
        f.write(df_latex.to_latex(index=True,
                                  index_names=True,
                                  escape=False,
                                  float_format="%.15g"))

    # 8) „Rounded“-LaTeX-Tabelle: a,b als k*pi^2-Strings; c,v0_first mit 6 Nachkommastellen
    rounded_cols = ['a_quad_tex', 'b_quad_tex', 'c_quad', 'v0_first']
    rounded_csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.csv")
    rounded_tex_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.tex")

    if os.path.exists(rounded_csv_path):
        rounded_df = pd.read_csv(rounded_csv_path, index_col='l')
    else:
        rounded_df = None
    rounded_df = _ensure_columns(rounded_df, rounded_cols, index_name='l')

    rounded_df.loc[l_key] = [
        latex_ab['a_quad'],
        latex_ab['b_quad'],
        float(c_value),
        float(v0_first)
    ]

    try:
        rounded_df.index = rounded_df.index.astype(int)
        rounded_df.sort_index(inplace=True)
    except Exception:
        pass
    rounded_df.to_csv(rounded_csv_path)

    latex_headers_rounded = {
        'a_quad_tex': r'$a_{\mathrm{quad}}$',
        'b_quad_tex': r'$b_{\mathrm{quad}}$',
        'c_quad': r'$c_{\mathrm{quad}}$',
        'v0_first': r'$v_0^{(1)}$'
    }
    rounded_df_latex = rounded_df.rename(columns=latex_headers_rounded)

    # Nur c_quad und v0_first formatieren (6 Nachkommastellen); a,b sind Strings ($k\pi^2$)
    formatters = {
        r'$c_{\mathrm{quad}}$':  safe_fmt_6,
        r'$v_0^{(1)}$':          safe_fmt_6
    }
    with open(rounded_tex_path, 'w') as f:
        f.write(rounded_df_latex.to_latex(index=True,
                                          index_names=True,
                                          escape=False,
                                          formatters=formatters))

    # Rückgabe: Fit-Funktion
    return None, (lambda idx: coeffs2[0]*idx**2 + coeffs2[1]*idx + coeffs2[2])


# ---- Beispiel-Runner (beide Typen, l=0..10) ----
if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "Prognosefunktionen")
    # Optional: alte rounded-Tabellen löschen, damit alles frisch ist
    for typ in ("Boundstates", "Resonanzen"):
        safe_typ  = typ.replace(" ", "")
        safe_kind = "limk0"
        rounded_csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.csv")
        rounded_tex_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.tex")
        if os.path.exists(rounded_csv_path): os.remove(rounded_csv_path)
        if os.path.exists(rounded_tex_path): os.remove(rounded_tex_path)

    for j in range(11):
        v0_forecast_func("Resonanzen", j, kind='lim_k_0')
        v0_forecast_func("Boundstates", j, kind='lim_k_0')

#
# v0_forcast_func_any_y("Resonanzen", 4, 5.0881106862211904)


def plot_v0s_multi_ordnungen(ordnungen, first_v0s, save=True):
    '''
    !!!Noch nicht basierend auf Resonanztabellen nur BS bis jetzt!!!
    mehrere Ordnungen in einem Plot übereinander legen (inklusive separater fits)
    '''

    plt.figure(figsize=(14, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ordnungen)))
    linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,2))]
    
    all_handles = []
    all_labels = []
    
    for i, l in enumerate(ordnungen):
        ordner = f"tables/l={l}"
        dateiname = os.path.join(ordner, f"neue_Boundstates_l={l}.csv")

        # Datei einlesen, Kopfzeile überspringen
        data = np.loadtxt(dateiname, delimiter=",", skiprows=1)

        # Dritte Spalte extrahieren
        v0s = data[:200, 2]
        indices = np.arange(len(v0s) - 1) + 1
        indices2 = np.arange(len(v0s)) + 1
        dv0s = np.diff(v0s)
        
        # Linearer Fit auf Abstände
        coeffs = np.polyfit(indices, dv0s, deg=1)
        fit_vals = np.polyval(coeffs, indices)

        # Quadratischer Fit auf v0s
        coeffs2 = np.polyfit(indices2, v0s, deg=2)
        fit_vals2 = np.polyval(coeffs2, indices2)

        # Plot: Abstände und Fit
        h1, = plt.plot(indices, dv0s, 'x', color=colors[i], alpha=0.7, label=f'Δ$v_0$ (Ordnung {l})')
        h2, = plt.plot(indices, fit_vals, linestyle=linestyles[i%len(linestyles)], color=colors[i], 
                       label=f'Lin-Fit Δ$v_0$ (Ordnung {l})')
        # Plot: v0 Werte und quadr. Fit
        h3, = plt.plot(indices2, v0s, 'o', color=colors[i], fillstyle='none', alpha=0.5, 
                       label=f'$v_0$ (Ordnung {l})')
        h4, = plt.plot(indices2, fit_vals2, linestyle=linestyles[i%len(linestyles)], color=colors[i], 
                       linewidth=1.8, alpha=0.5, label=f'Quad-Fit $v_0$ (Ordnung {l})')
        
        # Legendenhandle sammeln, aber nur einmal pro Kurventyp
        all_handles += [h1, h2, h3, h4]
        all_labels += [f'Δ$v_0$ (Ordnung {l})', f'Lin-Fit Δ$v_0$ (Ordnung {l})',
                       f'$v_0$ (Ordnung {l})', f'Quad-Fit $v_0$ (Ordnung {l})']
    
    plt.xlabel("# neuer Bound State")
    plt.ylabel("$v_0$")
    plt.title("Vergleich mehrerer Ordnungen")
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Platz rechts für die Legende lassen
    plt.legend(
        all_handles,
        all_labels,
        fontsize=9,
        ncol=1,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5)
    )
    plt.ylim(0, 100000)
    plt.xlim(0, 105)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    if save:
        save_dir = os.path.join(os.getcwd(), "Plots_Progosefunktion")
        os.makedirs(save_dir, exist_ok=True)
        ord_min = min(ordnungen)
        ord_max = max(ordnungen)
        filename = os.path.join(
            save_dir, 
            f"Vergleich_Ordnungen_{ord_min}-bis-{ord_max}.png"
        )
        plt.savefig(filename, dpi=200)

#plot_v0s_multi_ordnungen(list(range(0,10)), first_v0s)


#nicht richtig!!!!!!
def v0_forecast_global(max_l):
    # Alle V0-Werte einsammeln
    all_v0s = []
    for l in range(max_l + 1):
        v0s_l = np.array(first_v0s(l))
        all_v0s.extend(v0s_l)
    
    # Sortieren, falls nicht streng aufsteigend
    all_v0s = np.sort(np.array(all_v0s))
    indices = np.arange(len(all_v0s))
    dv0s = np.diff(all_v0s)
    indices_dv0 = np.arange(len(dv0s))
    
    # Quadratischer Fit auf alle V0-Werte
    coeffs_quad = np.polyfit(indices, all_v0s, deg=2)
    fit_v0s = np.polyval(coeffs_quad, indices)
    chi2_quad = np.sum((all_v0s - fit_v0s)**2)
    dof_quad = len(indices) - 3
    chi2_quad_red = chi2_quad / dof_quad if dof_quad > 0 else np.nan

    # Linearer Fit auf alle Deltas
    coeffs_lin = np.polyfit(indices_dv0, dv0s, deg=1)
    fit_dv0s = np.polyval(coeffs_lin, indices_dv0)
    chi2_lin = np.sum((dv0s - fit_dv0s)**2)
    dof_lin = len(indices_dv0) - 2
    chi2_lin_red = chi2_lin / dof_lin if dof_lin > 0 else np.nan

    # Funktionen zum Zurückgeben
    def v0_parabola(idx):
        return coeffs_quad[0]*idx**2 + coeffs_quad[1]*idx + coeffs_quad[2]
    def v0_step(idx):
        return coeffs_lin[0]*idx + coeffs_lin[1]

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(indices, all_v0s, 'x', label='$v_0$ Werte global')
    plt.plot(indices, fit_v0s, '--', label=(
        f'Globaler quadratischer Fit:\n'
        f'$a$={coeffs_quad[0]:.4g}, $b$={coeffs_quad[1]:.4g}, $c$={coeffs_quad[2]:.4g}\n'
        f'$\chi^2_{{red}}$={chi2_quad_red:.2g}'
    ))
    plt.plot(indices_dv0, dv0s, 'x', label='Abstände (Δ$v_0$) global')
    plt.plot(indices_dv0, fit_dv0s, '-', label=(
        f'Globaler linearer Fit der Abstände:\n'
        f'$a$={coeffs_lin[0]:.4g}, $b$={coeffs_lin[1]:.4g}\n'
        f'$\chi^2_{{red}}$={chi2_lin_red:.2g}'
    ))
    plt.xlabel("globaler Index / # aller gefundenen Bound States")
    plt.ylabel("$v_0$")
    plt.title(f"Globaler Fit aller Bound-State $v_0$ bis l={max_l}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Speicherpfad bauen
    save_dir = os.path.join(os.getcwd(), "Plots_Progosefunktion")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"Prognosefunktion_Global_l_max={max_l}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    
    return v0_parabola, v0_step

#v0_forecast_global(0)
#v0_forecast_global(1)
#v0_forecast_global(2)
#v0_forecast_global(3)
#v0_forecast_global(4)
#v0_forecast_global(5)
#v0_forecast_global(6)
#v0_forecast_global(7)
#v0_forecast_global(8)
#v0_forecast_global(9)