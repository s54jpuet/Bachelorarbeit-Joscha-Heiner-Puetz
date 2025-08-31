import os
import os.path
import numpy as np
import pandas as pd
import tableReading as tR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def ensure_columns(df, cols, index_name='l'):
    """Überprüfe korrekte Spaltenstruktur für input df hinsichtlich Spalten cols"""
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
    """Formatiere Zahlen auf 6 Nachkommastellen, Strings bleiben unverändert"""
    try:
        return f"{float(x):.6f}"
    except (ValueError, TypeError):
        return str(x) if pd.notna(x) else ""

# ---- Hauptfunktion ----
def v0_forecast_func(typ, l, kind="lim_k_0"):
    """
    Quadratischer Fit an Werte für die Prognosefunktion aufgestellt werden soll, typ als Resonanzen oder Boundstates und kind für y Wert, für den die Prognosefunktion aufgestellt werden soll (auch für minimalen v0 Wert der Resonanzen möglich mit min_v0)
    - Plot: Datenpunkte und Fit
    - LaTeX Tabelle mit gerundeten Fitparametern
    """
    # Finden der Inputdatei und Titelfestlegung abhängig von kind
    ordner = f"tables/l={l}"
    if kind == "lim_k_0":
        infile = os.path.join(ordner, f"neue_{typ}_l={l}_mit_Fehler.csv")
        plot_title = f"Prognosefunktion lim_k_0 {typ} l={l}"
    elif kind == "min_v0":
        infile = os.path.join(ordner, f"min_v0_{typ}_l={l}.csv")
        plot_title = f"Prognosefunktion min_v0 {typ} l={l}"
    else:
        raise ValueError(f"Unbekanntes kind='{kind}'")

    # Daten einlesen
    data = np.loadtxt(infile, delimiter=",", skiprows=1)
    v0s = data[:200, 2].astype(float)
    indices2 = np.arange(1, len(v0s) + 1, dtype=float)

    # Erzeugung der Fitfunktion ax**2+bx+c
    coeffs2 = np.polyfit(indices2, v0s, deg=2) 
    y_fit = np.polyval(coeffs2, indices2)

    v0_first = v0s[0]
    c_value  = coeffs2[2]

    # a,b als Vielfache von pi**^2 (für gerundete-Tabellen (in Bachelorarbeit nur für l=0,1 genutzt))
    pi2 = np.pi**2
    latex_ab = {
        'a_quad': rf"${int(np.round(coeffs2[0] / pi2))}\pi^2$",
        'b_quad': rf"${int(np.round(coeffs2[1] / pi2))}\pi^2$"
    }

    # Erzeugung Plot
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

    # Tabellen der Fitparameter (auch als Latex)
    safe_typ  = typ.replace(" ", "")
    safe_kind = kind.replace("_","")
    csv_path  = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.csv")
    tex_path  = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.tex")

    main_cols = ['a_quad', 'b_quad', 'c_quad', 'v0_first']
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='l')
    else:
        df = None
    df = ensure_columns(df, main_cols, index_name='l')

    df.loc[l] = [float(coeffs2[0]), float(coeffs2[1]), c_value, v0_first]

    # sortiere bei ganzzahligem Index
    try:
        df.index = df.index.astype(int)
        df.sort_index(inplace=True)
    except Exception:
        pass
    df.to_csv(csv_path)

    latex_headers = {
        'a_quad': r'$a$',
        'b_quad': r'$b_{\ell}$',
        'c_quad': r'$c_{\ell}$',
        'v0_first': r'$v_0^{(1)}$'
    }
    df_latex = df.rename(columns=latex_headers)
    with open(tex_path, 'w') as f:
        f.write(df_latex.to_latex(index=True,
                                  index_names=True,
                                  escape=False,
                                  float_format="%.15g"))

    # gerundete-Latex-Tabelle
    rounded_cols = ['a_quad_tex', 'b_quad_tex', 'c_quad', 'v0_first']
    rounded_csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.csv")
    rounded_tex_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.tex")

    if os.path.exists(rounded_csv_path):
        rounded_df = pd.read_csv(rounded_csv_path, index_col='l')
    else:
        rounded_df = None
    rounded_df = ensure_columns(rounded_df, rounded_cols, index_name='l')

    rounded_df.loc[l] = [
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

    # Nur c_quad und v0_first formatieren (6 Nachkommastellen), a,b sind Strings ($k\pi^2$)
    formatters = {
        r'$c_{\mathrm{quad}}$':  safe_fmt_6,
        r'$v_0^{(1)}$':          safe_fmt_6
    }
    with open(rounded_tex_path, 'w') as f:
        f.write(rounded_df_latex.to_latex(index=True,
                                          index_names=True,
                                          escape=False,
                                          formatters=formatters))

    return None, (lambda idx: coeffs2[0]*idx**2 + coeffs2[1]*idx + coeffs2[2])


if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "Prognosefunktionen")
    for typ in ("Boundstates", "Resonanzen"):
        safe_typ  = typ.replace(" ", "")
        safe_kind = "limk0"
        rounded_csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.csv")
        rounded_tex_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}_rounded.tex")
        if os.path.exists(rounded_csv_path): os.remove(rounded_csv_path)
        if os.path.exists(rounded_tex_path): os.remove(rounded_tex_path)

    # Funktionsaufruf für alle untersuchten Ordnungen
    for j in range(11):
        v0_forecast_func("Resonanzen", j, kind='lim_k_0')
        v0_forecast_func("Boundstates", j, kind='lim_k_0')




def eval_c_of_forecast_for_all_l(typ):
    '''Nicht in der Bachelorarbeit explizit genutzt aber möglicherweise ebenfalls interessant: Hier kann ein ordnungsübergreifender Polynomfit für den Koeffizienten c aus den quadratischen Prognosefunktionen für die berechneten Ordnungen l erzeugt werden'''
    save_dir = os.path.join(os.getcwd(), "Prognosefunktionen")
    safe_typ = typ.replace(" ", "")
    safe_kind = "limk0"
    csv_path = os.path.join(save_dir, f"forecast_coeffs_{safe_typ}_{safe_kind}.csv")
    df = pd.read_csv(csv_path, index_col='l')


    l_vals = pd.to_numeric(df.index, errors='coerce').to_numpy(dtype=float)
    c = pd.to_numeric(df.get('c_quad', pd.Series(index=df.index, dtype=float)), errors='coerce').to_numpy(dtype=float)

    mask = np.isfinite(l_vals) & np.isfinite(c)
    l_vals = l_vals[mask]
    c = c[mask]

    if l_vals.size < 2:
        return 0.0, 0.0, 0.0, float(np.nan)

    deg = min(3, int(l_vals.size) - 1)
    coef = np.polyfit(l_vals, c, deg=deg) 

    coef_full = np.zeros(4, dtype=float)
    coef_full[-(deg+1):] = coef
    a3, b3, n3, m3 = coef_full  
    try:
        c_fit = np.polyval(coef_full, l_vals)
        from sklearn.metrics import r2_score
        r2_3 = r2_score(c, c_fit) if l_vals.size >= 2 else float('nan')
    except Exception:
        r2_3 = float('nan')

    return a3, b3, n3, m3

#eval_c_of_forecast_for_all_l("Resonanzen")