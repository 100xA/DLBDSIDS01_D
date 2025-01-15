# DLBDSIDS01_D - Kundensegmentierung

## Projektübersicht
Dieses Projekt implementiert eine fortgeschrittene Kundensegmentierung unter Verwendung von K-means Clustering und PCA (Principal Component Analysis). Es analysiert demografische und Verhaltensdaten von Kunden, um aussagekräftige Segmente zu identifizieren und zu visualisieren.

## Hauptfunktionen
- **Datenvorverarbeitung**: Automatische Behandlung fehlender Werte und Standardisierung
- **K-means Clustering**: Identifikation von 5 distinkten Kundensegmenten
- **PCA-Analyse**: Dimensionsreduktion und Interpretation der Hauptkomponenten
- **Visualisierung**: Interaktive Plots zur Darstellung der Kundensegmente
- **Cluster-Validierung**: Berechnung verschiedener Validierungsmetriken

## Identifizierte Kundensegmente
1. **Familien mit mittlerem Einkommen** (24.2%)
   - Mittleres Alter, größere Familien
   - Überwiegend mittleres Ausgabenniveau

2. **Kostenbewusste Rentner** (24.5%)
   - Ältere Generation, kleine Haushalte
   - Sehr kostenbewusst

3. **Karriereorientierte Paare** (18.4%)
   - Hohe Berufserfahrung
   - Ausgewogenes Ausgabenverhalten

4. **Junge Großfamilien** (22.9%)
   - Jüngste Altersgruppe
   - Größte Familien, stark preisbewusst

5. **Wohlhabende Best Ager** (10.0%)
   - Ältere, wohlhabende Zielgruppe
   - Höchster Anteil an hohen Ausgaben

## Projektstruktur
```
DLBDSIDS01_D/
├── data/
│   └── raw/                  # Original data
├── notebooks/                # Jupyter notebooks
├── output/
│   └── figures/              # Generated plots
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation & Ausführung
1. Repository klonen:
   ```bash
   git clone [repository-url]
   cd DLBDSIDS01_D
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix
   # oder
   .\venv\Scripts\activate  # Windows
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Skript ausführen:
   ```bash
   python src/customer_segmentation/main.py
   ```

## Technische Details
- **Sprache**: Python 3.8+
- **Hauptbibliotheken**: 
  - scikit-learn (Clustering, PCA)
  - pandas (Datenverarbeitung)
  - matplotlib, seaborn (Visualisierung)
  - numpy (Numerische Operationen)

## Validierungsmetriken
- Silhouette Score: 0.326
- Calinski-Harabasz Index: 3284.2
- Davies-Bouldin Index: 1.073
- Durchschnittliche Zentroid-Separation: 2.608

## PCA-Interpretation
- **PC1** (37.7% Varianz): Alter und Ausgabenniveau
- **PC2** (27.9% Varianz): Familiengröße und Arbeitserfahrung

## Lizenz
MIT License
