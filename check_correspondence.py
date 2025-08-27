import os
import os.path
import numpy as np
import pandas as pd
import tableReading as tR


def check_correspondence(l):
    '''Diese Funktion gleicht Paarweise die Werte für neuauftretende Resonanzen und gebundene Zustände ab und überprüft deren Übereinstimmung im Rahmen des gemeinsamen Fehlers'''
    ordner = f"tables/l={l}"
    infile_rs = os.path.join(ordner, f"neue_Resonanzen_l={l}_mit_Fehler.csv")
    infile_bs = os.path.join(ordner, f"neue_Boundstates_l={l}_mit_Fehler.csv")

    # 2) Daten einlesen
    data_rs = np.loadtxt(infile_rs, delimiter=",", skiprows=1)
    data_bs = np.loadtxt(infile_bs, delimiter=",", skiprows=1)

    v0_rs = data_rs[:, 2].astype(float)
    v0_bs = data_bs[:, 2].astype(float)

    if data_rs.shape[1] < 4:
        raise ValueError("Erwartete 4 Spalten (inkl. sigma_v0) für lim_k_0, aber weniger gefunden.")
    
    # 3) Spalten holen
    v0_rs = data_rs[:, 2].astype(float)
    s_rs  = data_rs[:, 3].astype(float)

    v0_bs = data_bs[:, 2].astype(float)
    s_bs  = data_bs[:, 3].astype(float)

    # 4) auf gemeinsame Länge bringen (Sicherheit)
    n = min(v0_rs.size, v0_bs.size)
    v0_rs, s_rs = v0_rs[:n], s_rs[:n]
    v0_bs, s_bs = v0_bs[:n], s_bs[:n]

    # Differenz + kombinierter Fehler
    delta = v0_bs - v0_rs
    sigma_delta = s_rs + s_bs 

    # Prüfen, ob |delta| ≤ sigma_delta
    within = np.abs(delta) <= 10 * sigma_delta
    anzahl_within = np.sum(within)

    print(f"[l={l}] Punkte: {n}, innerhalb gemeinsamer Fehlergrenze: {anzahl_within}/{n}, max|delta|={np.max(np.abs(delta)):.3g}")

    out_dir = "Prognosefunktionen"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"Differenzen_l={l}.csv")
    header = "index,v0_bs,v0_rs,delta,sigma_bs,sigma_rs,sigma_delta,within_error"
    idx = np.arange(1, n+1, dtype=int)
    out = np.column_stack([idx, v0_bs, v0_rs, delta, s_bs, s_rs, sigma_delta, within])
    np.savetxt(out_csv, out, delimiter=",", header=header, comments='', fmt="%.18e")
    print(f"[l={l}] Zeilenweise Differenzen gespeichert: {out_csv}")

    return {
        "l": l,
        "n": n,
        "within_error_count": int(anzahl_within),
        "csv": out_csv,
    }

# Aufruf aller berechneten Ordnungen zur Überprüfung derer Korrespondenz:
j = 0
while j < 11:
    check_correspondence(j)
    j += 1