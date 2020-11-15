### Setup the venv
````
virtualenv -p python3.6 venv
source venv/bin/activate

pip install -r requirements.txt
````

### Aufbau des Codes

#### Vorbereitung der Daten
In `data_preparation.py` werden die Daten eingelesen, in einer Datenstruktur gespeichert und diese serialisiert. Die
 aufwändigen Berechnungen wie die R-Peak-Detection und die Anwendung des CLIE-Algorithmus werden unabhängig von dieser 
 Datenstruktur in csv-Dateien serialisiert.
 
 #### Berechnung der Merkmale
 Die Merkmale für die verschiedenen Verfahren werden in `data_statistical_features.py` berechnet und serialisiert.
 
 #### Implementierung und Training der Modelle
 Die Modelle selbst sind in `estimators.py` implementiert. Die selbst entwickelten Modelle finden sich zusätzlich noch 
 in `own_models.py`.
 
 
 #### Reproduktion der Ergebnisse
 Eine einfache Reproduktion der Ergebnisse kann durch das Ausführen des Skriptes `reproduce_results.py` erreicht werden.
