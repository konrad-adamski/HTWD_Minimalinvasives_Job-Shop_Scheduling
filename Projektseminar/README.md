
##  Makespan

Zum Testen starten Sie: ```00_cp_makespan.py```

##  Minimalinvasives Scheduling

Führen Sie im Terminal folgende Befehle aus:


```cmd
python 01_run_minimalinvasive_cp_experiment.py --util 0.80 --lateness_ratio 1.0 --tardiness_ratio 0.5 --sim_sigma 0.15    
```

```cmd
python 01_run_minimalinvasive_cp_experiment.py --util 0.80 --lateness_ratio 0.5 --tardiness_ratio 0.5 --sim_sigma 0.15    
```

```cmd
python 01_run_minimalinvasive_cp_experiment.py --util 0.80 --lateness_ratio 0.5 --tardiness_ratio 1.0 --sim_sigma 0.15    
```

##  Einblick in die Ergebnisse

Im Ordner ```output``` finden Sie die Resultate als CSV usw., zusätzlich können Sie gerne einen Blick in die ```experimets.db``` im Root-Verzeichnis werfen.

Eine kleine Auswertung ist zudem über ```python 01b_evaluation.py``` verfügbar.


<br>
(Beim OSError überprüfen Sie, ob Sie sich im Projektseminar-Verzeichnis befinden!)


