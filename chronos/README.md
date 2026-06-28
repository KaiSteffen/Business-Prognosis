<div align="center">

# 🔮 Business-Prognosis · Chronos

### Zeitreihen-Prognose mit Chronos (Foundation Model)

![Hochschule Aalen](https://img.shields.io/badge/Hochschule%20Aalen-SS%202026-8A2BE2)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Chronos](https://img.shields.io/badge/Forecasting-Chronos-FF9900)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

*Studienprojekt an der Hochschule Aalen, Sommersemester 2026*

</div>

---

## 📌 Über das Projekt

Dieses Teilprojekt überträgt das **Prophet**-Material auf **Chronos** – ein
**vortrainiertes Foundation Model** für Zeitreihen (Amazon Science). Wie beim
Prophet-Teil entstehen **Präsentation**, **Demo** und **Tutorial**.

Der spannende Kontrast: Prophet ist „Analyst-in-the-Loop" (du baust Feiertage,
Regressoren etc. selbst ein), Chronos sagt **zero-shot** vorher – ganz **ohne Training
und ohne Feature-Engineering**. Beides wird auf **denselben Rossmann-Daten** gezeigt, um
direkt vergleichen zu können.

---

## 📈 Demo: dieselben Daten wie Prophet

Am Beispiel der **Rossmann-Filiale 1097** wird vom naiven Zero-Shot-Forecast zu einer
optimierten Vorhersage gearbeitet – Notebook
[`1_chronos_demo.ipynb`](notebooks/1_chronos_demo.ipynb):

| Schritt | Inhalt | Prophet-Gegenstück |
|:---:|:---|:---|
| 1 | Zero-Shot Baseline (Bolt-small) | Baseline (Trend+Saison) |
| 2 | Kontextlänge variieren | + Feiertage |
| 3 | Modellgröße vergleichen | + regionale Feiertage/Zeitfenster |
| 4 | Kovariate „Promo" (Chronos-2) | + Regressor |
| 5 | Prognoseintervalle (Quantile) | Unsicherheit |

> 🦢 **Black-Swan-Stresstest** mit den Flugdaten – derselbe Test wie bei Prophet.

---

## 🎓 Hands-On für Studierende

[`2_chronos_handson_students.ipynb`](notebooks/2_chronos_handson_students.ipynb) –
Crashkurs zum Selbermachen mit dem „SkyDrive X1 Pro"-Datensatz:

1. **Aufgabe 1** – naiver Zero-Shot-Forecast mit Chronos-Bolt
2. **Aufgabe 2** – Modell verbessern (mehr Kontext/größeres Modell, Bonus: Black Friday
   als Kovariate mit Chronos-2)

Musterlösung in
[`3_chronos_handson_solution.ipynb`](notebooks/3_chronos_handson_solution.ipynb).

---

## 🗂️ Projektstruktur

```
.
├── data/
│   ├── rossmann-store-sales/   # Rossmann-Datensatz (identisch zum Prophet-Teil)
│   └── flights.csv             # Flugdaten für den Black-Swan-Stresstest
├── notebooks/
│   ├── 1_chronos_demo.ipynb             # Demo (Rossmann)
│   ├── 2_chronos_handson_students.ipynb # Übung für Studierende
│   └── 3_chronos_handson_solution.ipynb # Musterlösung
├── presentation/
│   └── Chronos_Praesentation.pptx       # Einstiegspräsentation
├── literature/                          # Chronos-Paper (PDFs folgen)
├── ts-tutorial.yml                      # Conda-Umgebung (+ Chronos)
└── README.md
```

---

## ⚙️ Setup

Chronos baut auf der bestehenden Prophet-Umgebung (`ts-tutorial`) auf und ergänzt nur
die Chronos-Pakete. **Empfohlen:** in die vorhandene Env installieren –

```bash
conda activate ts-tutorial
pip install chronos-forecasting scikit-learn "pandas[pyarrow]"
```

`pip install chronos-forecasting` zieht `transformers`, `accelerate` und `einops`
automatisch mit; `pandas[pyarrow]` wird nur für den optionalen Chronos-2-Teil gebraucht.

Alternativ lässt sich die Umgebung komplett aus der mitgelieferten Datei nachbauen
(sie enthält Prophet **und** Chronos):

```bash
conda env create -f ts-tutorial.yml --no-channel-priority
conda activate ts-tutorial
jupyter lab
```

**Wichtigste Pakete:** Python 3.10 · chronos-forecasting · PyTorch (CPU) · Pandas ·
NumPy · Matplotlib · JupyterLab

> Beim ersten `from_pretrained(...)` lädt das Modell automatisch von Hugging Face
> (einmalig Internet nötig, danach im Cache). Die empfohlenen **Bolt-small/base**-Modelle
> laufen auf der **CPU** in Sekunden.

---

## 📊 Daten

- **Rossmann Store Sales** – tägliche Umsätze einzelner Filialen (Kaggle-Datensatz)
- **Flugdaten** – ergänzender Datensatz für den Strukturbruch-/Black-Swan-Stresstest

---

## 📚 Literatur

Im Ordner `literature/` liegen die zugrundeliegenden Paper (PDFs folgen):

- **Ansari, A. F. et al. (2024). *Chronos: Learning the Language of Time Series.*** TMLR.
  [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
- **Ansari, A. F. et al. (2025). *Chronos-2: From Univariate to Universal Forecasting.***
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)

---

<div align="center">

**Bettina & Kai** — Hochschule Aalen · SS 2026

</div>
