<div align="center">

# 🔮 Business-Prognosis

### Zeitreihen-Prognose mit Prophet

<img src="images/prophet_wizard.png" alt="Prophet – Time Series Forecasting" width="420" />

<br/>

<img src="images/prophet_team.png" alt="Prophet – Business Forecasting Team" width="300" />

<br/>

![Hochschule Aalen](https://img.shields.io/badge/Hochschule%20Aalen-SS%202026-8A2BE2)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-4B0082)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

*Studienprojekt an der Hochschule Aalen, Sommersemester 2026*

</div>

---

## 📌 Über das Projekt

Ziel des Projekts ist es, das Forecasting-Verfahren **Prophet** am Beispiel realer
Verkaufszahlen (Rossmann-Filialen) praktisch zu erproben und für andere Studierende
aufzubereiten – als **Präsentation**, **Demo** und **Tutorial**.

Prophet (Facebook/Meta) ist ein additives Modell aus **Trend**, **Saisonalität** und
**Feiertagen**. Der Reiz: Es ist robust, interpretierbar und lässt sich vom Analysten
gezielt mit Domänenwissen verbessern.

---

## 📈 Prophet-Tutorial: „Analyst-in-the-Loop"

Am Beispiel der **Rossmann-Filiale 1097 (Kölner Hauptbahnhof)** wird Schritt für Schritt
vom einfachen Baseline-Modell zu einer optimierten Vorhersage gearbeitet:

| Schritt | Modell | Erweiterung |
|:---:|:---|:---|
| 1 | Baseline | Trend + Saisonalität |
| 2 | + Feiertage | bundesweite deutsche Feiertage |
| 3 | + Regionales | NRW-Feiertage + Karneval |
| 4 | + Zeitfenster | Wirkungsfenster rund um Feiertage |
| 5 | + Regressor | Promotionen als zusätzliche Einflussgröße |

> 🦢 **Black-Swan-Stresstest:** Ein Zusatzkapitel zeigt, wo Prophet an Grenzen stößt,
> wenn die gelernte Saisonalität auf einen plötzlichen Strukturbruch in den Daten trifft.

---

## 🎓 Hands-On für Studierende

Das Notebook [`prophet_handson_students.ipynb`](notebooks/prophet_handson_students.ipynb)
ist ein **Crashkurs zum Selbermachen**. Anhand eines Beispiel-Datensatzes (Verkäufe der
„SkyDrive X1 Pro" SSD) lösen die Studierenden zwei Aufgaben:

1. **Aufgabe 1** – ein erstes Prophet-Modell *ohne* Feiertage trainieren („naives Modell")
2. **Aufgabe 2** – das Modell um den **Black Friday** als Feiertag erweitern („Experten-Modell")

Setup, Datengenerierung und Visualisierungen sind vorgegeben (einfach ausführen), sodass
der Fokus auf dem Modellieren liegt. Ein direkter Vergleich (naiv vs. Experte) und
Diskussionsfragen runden die Übung ab. Die vollständige Musterlösung liegt in
[`prophet_handson_solution.ipynb`](notebooks/prophet_handson_solution.ipynb).

---

## 🗂️ Projektstruktur

```
.
├── data/                 # Datensätze (Rossmann Store Sales, Flugdaten)
├── notebooks/            # Jupyter-Notebooks
│   ├── prophet_demo_final.ipynb         # vollständige Demo
│   ├── prophet_handson_students.ipynb   # Übung für Studierende
│   └── prophet_handson_solution.ipynb   # Musterlösung
├── powerpoint/           # Einstiegspräsentation
├── literature/           # Wissenschaftliche Quellen (PDFs)
├── images/               # Key Visuals
├── ts-tutorial.yml       # Conda-Umgebung
└── README.md
```

---

## ⚙️ Setup

Die benötigte Umgebung lässt sich mit Conda erstellen:

```bash
conda env create -f ts-tutorial.yml
conda activate ts-tutorial
jupyter lab
```

**Wichtigste Pakete:** Python 3.10 · Prophet · Pandas · NumPy · Matplotlib · PyTorch · JupyterLab

---

## 📊 Daten

- **Rossmann Store Sales** – tägliche Umsätze einzelner Filialen (Kaggle-Datensatz)
- **Flugdaten** – ergänzender Datensatz für den Strukturbruch-/Black-Swan-Stresstest

---

## 📚 Literatur

Im Ordner `literature/` liegen die zugrundeliegenden Paper, u. a.:

- Taylor, S. J. & Letham, B. (2018). *Forecasting at Scale.* The American Statistician 72(1), 37–45.
- Triebe et al. (2021). *NeuralProphet – Explainable Forecasting at Scale.*
- Žunić, E. et al. (2020). *Application of Facebook's Prophet Algorithm.*

---

<div align="center">

## 👥 Team

**Bettina & Kai** — Hochschule Aalen · SS 2026

</div>
