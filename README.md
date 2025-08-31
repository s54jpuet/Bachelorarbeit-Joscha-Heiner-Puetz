# Vertiefte Betrachtungen zum Zusammenhang zwischen Resonanzen in der Streuung und der Existenz von gebundenen Zuständen

Dieses Repository enthält den vollständigen, kommentierten Quellcode sowie alle zur Auswertung verwendeten Tabellen und Plots der Bachelorarbeit  

**„Vertiefte Betrachtungen zum Zusammenhang zwischen Resonanzen in der Streuung und der Existenz von gebundenen Zuständen“**  
eingereicht am Physikalischen Institut der Rheinischen Friedrich-Wilhelms-Universität Bonn (2025).

---

## Inhalt

Die Arbeit untersucht systematisch den Zusammenhang zwischen **Resonanzen in der elastischen Streuung** und der **Existenz gebundener Zustände** im Rahmen des **sphärisch symmetrischen Kastenpotentials**.  
Grundlage bilden die Bedingungen aus der Streutheorie, welche sich mittels Partialwellenzerlegung formulieren lassen und für kleine Drehimpulsordnungen analytisch überprüfbar sind. Für Ordnungen bis ℓ = 10 wurden die Lösungen numerisch bestimmt.  

**Zentrale Ergebnisse:**
- Im Niederenergielimit besteht eine **vollständige Korrespondenz** zwischen Resonanzen und gebundenen Zuständen.  
- Unterschiede ergeben sich lediglich im **Verlauf der Resonanzspuren**: monoton für ℓ = 0, nicht-monoton für ℓ > 0 aufgrund der Zentrifugalbarriere.  
- Die Schwellenwerte für neu auftretende Zustände lassen sich durch **einfache quadratische Prognosefunktionen** beschreiben, die numerisch bestätigt und analytisch hergeleitet werden konnten.  
- Damit ergibt sich ein direkter Bezug zur Interpretation realer Streuexperimente (z. B. Neutron-Proton-Streuung).

---

## Reproduzierbarkeit

Alle in der Arbeit dargestellten Ergebnisse sind mit diesem Repository vollständig reproduzierbar.  

- **Programmiersprache:** Python 3.11.9  
- **Abhängigkeiten:** NumPy, SciPy, Matplotlib, Pandas, scikit-learn (siehe `requirements.txt`)  
- **Ausführung:**  
  - Alle Berechnungen können direkt über die Python-Skripte gestartet werden.  
  - Zusätzlich stehen Jupyter-Notebooks (`CreateTables.ipynb`, `WellenfunktionenPlots.ipynb`) für eine strukturierte Analyse und Visualisierung zur Verfügung.  

---

## Struktur des Repositories

- `Bessel.py` – Implementierung sphärischer Bessel-, Neumann- und Hankel-Funktionen inkl. Ableitungen  
- `resonanzen.py` – Bedingungen und Berechnung der Resonanzen sowie radiale Wellenfunktionen  
- `bound_states.py` – Bedingungen und Berechnung gebundener Zustände sowie radiale Wellenfunktionen  
- `roots_y.py` / `tracing.py` – Routinen zur Nullstellensuche und zur Verfolgung von Resonanzspuren  
- `tableReading.py` – Einlesen und Verarbeitung der erzeugten Tabellendateien  
- `CreateTables.ipynb` – zentrale numerische Analyse aller Resonanzen und Bound States (bis ℓ = 10)  
- `Create_Plots_for_tabels.py` – Generierung der in der Arbeit verwendeten Konturplots  
- `Heatmap_sin2.py` – Heatmaps und 3D-Plots des Sinusquadrats der Streuphase  
- `WellenfunktionenPlots.py` / `WellenfunktionenPlots.ipynb` – Darstellung der radialen Wellenfunktionen  
- `PrognoseFunktion.py` – Berechnung und Fit der Prognosefunktionen für Resonanzen und Bound States  
- `check_correspondence.py` – Punktweiser Abgleich von Resonanzen und gebundenen Zuständen im Niederenergielimit  
- `main.pdf` – vollständige Bachelorarbeit (schriftliche Ausarbeitung)  

Zusätzlich:
- **`tables/`** – enthält sämtliche numerischen Ergebnisse (Resonanzen und Bound States für alle ℓ ≤ 10).  
- **`Plots_*/`** – enthält alle in der Arbeit verwendeten Plots sowie zusätzliche Visualisierungen (Konturplots, Heatmaps, Wellenfunktionen).  

---

## Enthaltene Daten

Dieses Repository umfasst:
- **alle Tabellen**, die in der Bachelorarbeit zur Datenauswertung verwendet wurden,  
- **alle Plots**, die in der Arbeit dargestellt sind,  
- sowie **zusätzliche Plots**, die über die BA hinaus erstellt wurden.  

Damit sind sämtliche numerischen Resultate und deren Visualisierungen direkt zugänglich und reproduzierbar.

