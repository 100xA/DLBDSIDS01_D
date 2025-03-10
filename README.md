# DLBDSIDS01_D - Kundensegmentierung

## Projektübersicht
Dieses Projekt implementiert eine fortgeschrittene Kundensegmentierung unter Verwendung von K-means Clustering und PCA (Principal Component Analysis). Es analysiert demografische und Verhaltensdaten von Kunden, um aussagekräftige Segmente zu identifizieren und zu visualisieren.

## Hauptfunktionen
- **Datenvorverarbeitung**: Automatische Behandlung fehlender Werte und Standardisierung
- **K-means Clustering**: Identifikation von distinkten Kundensegmenten mit optimaler Clusteranzahl
- **PCA-Analyse**: Dimensionsreduktion und Interpretation der Hauptkomponenten
- **Visualisierung**: Interaktive 3D-Plots zur Darstellung der Kundensegmente
- **Cluster-Validierung**: Berechnung verschiedener Validierungsmetriken (Silhouette, Elbow)

## Projektstruktur
```
DLBDSIDS01_D/
├── data/
│   ├── raw/                  # Original Kundendaten
│   └── processed/            # Verarbeitete Daten mit Clusterzuordnungen
├── src/
│   ├── main.py               # Hauptskript zur Ausführung der Analyse
│   ├── preprocessing.py      # Datenvorverarbeitung und -bereinigung
│   ├── clustering.py         # K-means Clustering und Validierung
│   └── visualization.py      # PCA und Visualisierungsfunktionen
├── output/
│   └── figures/              # Generierte Visualisierungen
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation & Ausführung mit uv

[uv](https://github.com/astral-sh/uv) ist ein schneller, zuverlässiger Python-Paketmanager, der als Alternative zu pip und venv dient.

1. Repository klonen:
   ```bash
   git clone [repository-url]
   cd DLBDSIDS01_D
   ```

2. Installation von uv (falls noch nicht installiert):
   ```bash
   # Mit pip
   pip install uv
   
   # Oder mit Homebrew auf macOS
   brew install uv
   ```

3. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
   ```bash
   # Erstellt eine virtuelle Umgebung und installiert Abhängigkeiten
   uv venv
   uv pip install -r requirements.txt
   ```

4. Virtuelle Umgebung aktivieren:
   ```bash
   # Auf Unix/macOS
   source .venv/bin/activate
   
   # Auf Windows
   .venv\Scripts\activate
   ```

5. Skript ausführen:
   ```bash
   cd src
   uv run main.py
   ```

## Technische Details

### Datenvorverarbeitung (preprocessing.py)
- **Datenladung**: Einlesen der Kundendaten aus CSV-Dateien
- **Behandlung fehlender Werte**: Automatische Ersetzung fehlender Werte für Arbeitserfahrung und Familiengröße
- **Feature-Mapping**: Konvertierung kategorischer Werte (z.B. 'Low', 'Average', 'High') in numerische Werte
- **Standardisierung**: Normalisierung der Features mit StandardScaler

### Clustering (clustering.py)
- **Optimale Clusteranzahl**: Bestimmung der optimalen Anzahl von Clustern mittels Elbow-Methode und Silhouette-Score
- **K-means Clustering**: Anwendung des K-means Algorithmus zur Kundensegmentierung
- **Cluster-Analyse**: Detaillierte Analyse der demografischen Merkmale und des Ausgabenverhaltens jedes Clusters
- **Validierungsmetriken**:
  - Silhouette Score: Misst die Kohäsion innerhalb der Cluster und die Separation zwischen Clustern
  - Zentroid-Separation: Durchschnittliche Distanz zwischen Cluster-Zentren
  - Elbow Point: Optimaler Clusteranzahl (K)

### Visualisierung (visualization.py)
- **PCA**: Dimensionsreduktion zur Visualisierung der Kundensegmente in 3D
- **Komponenten-Analyse**: Interpretation der Hauptkomponenten und ihrer Ladungen
- **3D-Visualisierung**: Darstellung der Kundensegmente im PCA-Raum mit Cluster-Zentren
- **Feature-Importance**: Visualisierung der Bedeutung verschiedener Features basierend auf PCA-Ladungen

## Kundensegmente
Das Projekt identifiziert verschiedene Kundensegmente, darunter:
- **Premium Familien**: Kunden mit hohen Ausgaben und größeren Familien
- **Junge Sparfamilien**: Jüngere Kunden mit niedrigeren Ausgaben und Familien
- **Erfahrene Singles**: Ältere Kunden mit mittlerer bis hoher Arbeitserfahrung und kleiner Haushaltsgröße
- **Sparsame Senioren**: Ältere Kunden mit niedrigen Ausgaben
- **Mittlere Ausgeber**: Kunden mit durchschnittlichen Werten in allen Kategorien
- **Sparsame Singles**: Alleinstehende mit niedrigen Ausgaben
- **Erfahrene Ausgeber**: Kunden mit hoher Arbeitserfahrung und höheren Ausgaben
- **Junge Erfahrene**: Jüngere Kunden mit überraschend hoher Arbeitserfahrung
- **Wohlhabende Senioren**: Ältere Kunden mit hohen Ausgaben
- **Große Jungfamilien**: Jüngere Kunden mit größeren Familien

## PCA-Interpretation
- **PC1** (ca. 38% Varianz): Repräsentiert hauptsächlich Alter und Ausgabenniveau
- **PC2** (ca. 28% Varianz): Repräsentiert Familiengröße und Arbeitserfahrung
- **PC3** (ca. 20% Varianz): Kombiniert verschiedene demografische Faktoren

## Ergebnisse und Anwendungsfälle
Die identifizierten Kundensegmente können für verschiedene Geschäftsstrategien genutzt werden:
- **Zielgerichtetes Marketing**: Anpassung von Marketingkampagnen an spezifische Kundensegmente
- **Produktentwicklung**: Entwicklung neuer Produkte oder Dienstleistungen für bestimmte Segmente
- **Kundenbindung**: Entwicklung von Kundenbindungsstrategien basierend auf Segmentcharakteristiken
- **Umsatzoptimierung**: Identifikation von Segmenten mit hohem Umsatzpotenzial

## Abhängigkeiten
- numpy (>= 1.21.0)
- pandas (>= 1.3.0)
- scikit-learn (>= 1.0.0)
- matplotlib (>= 3.4.0)
- seaborn (>= 0.11.0)
- jupyter (>= 1.0.0)
- requests (== 2.31.0)
- openpyxl (== 3.1.2)
- kneed (für Elbow-Punkt-Detektion)

## Lizenz
MIT License
