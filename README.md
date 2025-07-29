# Minimalinvasives Job-Shop Scheduling


# 🧮 Projektsetup

Dieses Projekt nutzt verschiedene Python-Bibliotheken für Datenanalyse, Simulation und Optimierung. Unten findest du die Anweisungen zur Installation der Abhängigkeiten – jeweils für **Windows** und **Unix-basierte Systeme** (Linux, macOS).

---

## 🛠️ Installation

### 🔹 Voraussetzungen

- **Python 3.10 oder höher**
- **Aktuelle pip-Version**
- Optional: Verwendung einer **virtuellen Umgebung**

---

### 🪟 Installation unter Windows

```cmd
:: Virtuelle Umgebung erstellen (optional, empfohlen)
python -m venv venv
venv\Scripts\activate
```

```cmd
:: pip aktualisieren
python -m pip install --upgrade pip
```

```cmd
:: Pakete installieren
pip install pyyaml jupyter ipykernel pandas matplotlib seaborn simpy pulp ortools editdistance scipy sqlalchemy pydantic
```
---

### 🐧 Installation unter Linux / macOS

```bash
# Virtuelle Umgebung erstellen (optional, empfohlen)
python3 -m venv venv
source venv/bin/activate
```

```bash
# pip aktualisieren
python3 -m pip install --upgrade pip
```

```bash
# Pakete installieren
python3 -m pip install pyyaml jupyter ipykernel pandas matplotlib seaborn simpy pulp ortools editdistance scipy sqlalchemy pydantic
```