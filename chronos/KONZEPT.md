# 🧭 Konzept & Vorschlag – Chronos-Teilprojekt

*Gegenstück zum Prophet-Projekt · Hochschule Aalen · SS 2026*

Dieses Dokument ist der **Vorschlag**, wie wir das Prophet-Material 1:1 auf **Chronos**
übertragen. Es beschreibt die Ordnerstruktur, die drei Notebooks (Demo, Hands-on,
Lösung) und welche Daten wir nehmen. Der Code wird danach gemeinsam Schritt für Schritt
in die Gerüst-Notebooks eingefüllt.

---

## 1. Was ist Chronos – und wie unterscheidet es sich von Prophet?

Beide sagen Zeitreihen vorher, aber die Philosophie ist **gegensätzlich** – das ist der
rote Faden für die Präsentation:

| | **Prophet** | **Chronos** |
|---|---|---|
| Grundidee | Statistisches additives Modell (Trend + Saison + Feiertage) | **Foundation Model** (Transformer), auf Millionen Zeitreihen vortrainiert |
| Vorgehen | „Analyst-in-the-Loop" – du baust Domänenwissen ein (Feiertage, Promos …) | **Zero-Shot** – Modell bekommt nur die Historie und sagt vorher, *ohne* Training |
| Training nötig? | Ja, pro Zeitreihe ein `fit()` | Nein – das Modell ist fertig vortrainiert, du machst nur `predict` |
| Feature-Engineering | Zentral (Feiertage, Regressoren, Saisonalität konfigurieren) | Kaum – Modell erkennt Muster selbst; Kovariaten nur optional (Chronos-2) |
| Interpretierbarkeit | Hoch (Komponenten-Plots) | Niedrig (Black Box), dafür probabilistisch (Quantile) |
| Analogie | Ein Statistiker, dem du alles erklärst | Ein „GPT für Zeitreihen", das schon viel gesehen hat |

**Kernbotschaft fürs Tutorial:** Prophet = Wissen *hineinstecken*. Chronos = Wissen ist
*schon drin*. Genau diesen Kontrast machen wir auf denselben Rossmann-Daten sichtbar.

### Chronos-Modellfamilien (Stand: Paket-Version 2.x, Dez 2025)

- **Chronos-2** (`amazon/chronos-2`, 120M) – neuestes Modell, kann univariat,
  multivariat **und Kovariaten** (z. B. Promotionen) zero-shot. → unser Gegenstück zum
  „Regressor"-Schritt von Prophet.
- **Chronos-Bolt** (`amazon/chronos-bolt-tiny` 9M … `-base` 205M) – schnelle Variante,
  bis zu 250× schneller, gibt direkt Quantile aus. **Ideal für den Unterricht auf CPU.**
- **Chronos-T5** (`amazon/chronos-t5-tiny` 8M … `-large` 710M) – die Original-Variante,
  zieht Forecast-Stichproben aus einem Sprachmodell.

Für Demo & Hands-on empfehle ich **Chronos-Bolt (small/base)** auf der CPU – das läuft
auf Bettinas/deinem Laptop in Sekunden, genau wie die Prophet-Umgebung (PyTorch CPU).

---

## 2. Ordnerstruktur (Mirror zum Prophet-Projekt)

```
chronos/
├── data/
│   ├── rossmann-store-sales/   # identische Rossmann-Daten (train, test, store, …)
│   └── flights.csv             # Flugdaten für den Black-Swan-Stresstest
├── notebooks/
│   ├── 1_chronos_demo.ipynb            # vollständige Demo (Rossmann, Filiale 1097)
│   ├── 2_chronos_handson_students.ipynb # Übung für Studierende
│   └── 3_chronos_handson_solution.ipynb # Musterlösung
├── literature/                 # Chronos-Paper (siehe Abschnitt 6)
├── images/                     # Key Visuals
├── powerpoint/                 # Chronos-Einstiegspräsentation
├── chronos-tutorial.yml        # Conda-Umgebung (Prophet-Env + chronos-forecasting)
├── KONZEPT.md                  # dieses Dokument
├── .gitignore
└── README.md
```

> Die Daten sind bewusst **kopiert** (nicht verlinkt), damit `chronos/` eigenständig
> ist und die Notebook-Pfade exakt wie bei Prophet `../data/...` lauten.

---

## 3. Demo-Notebook `1_chronos_demo.ipynb` – Gliederung

**Gleiche Daten wie Prophet: Rossmann-Filiale 1097.** So entsteht ein direkter
1:1-Vergleich Prophet vs. Chronos. Die Schritte spiegeln die Prophet-Demo, übersetzt in
die Chronos-Denkweise:

| Schritt | Prophet-Original | Chronos-Gegenstück | Lernziel |
|:---:|:---|:---|:---|
| 0 | Setup & Metriken | Setup, `print_metrics()` wiederverwenden | gleiche Bewertung (RMSE, MAE, MAPE) |
| 1 | Daten laden (Filiale 1097) | identisch laden, in Chronos-Format bringen (Kontext + Horizont) | Datenaufbau verstehen |
| 2 | Baseline (Trend+Saison) | **Zero-Shot Bolt-small** ohne jede Konfiguration | „Es sagt sofort vorher – ohne fit!" |
| 3 | + Feiertage | **Kontextlänge variieren** (wie viel Historie sieht das Modell?) | wichtigster Chronos-Hebel statt Feiertage |
| 4 | + Regionales / Zeitfenster | **Modellgröße vergleichen** (tiny → base, ggf. Chronos-2) | Größe vs. Genauigkeit vs. Zeit |
| 5 | + Promo-Regressor | **Chronos-2 mit Kovariate „Promo"** (`future_df`) | echtes Gegenstück zum Regressor |
| 6 | Quantile/Unsicherheit (implizit) | **Prognoseintervalle** `quantile_levels=[0.1,0.5,0.9]` plotten | probabilistischer Forecast |
| 7 | Black-Swan-Stresstest (Flüge) | **gleicher Stresstest mit Chronos** | wo stößt ein Foundation Model an Grenzen? |
| 8 | Fazit | **Direktvergleich Prophet vs. Chronos** (eine Tabelle/ein Plot) | Stärken/Schwächen abwägen |

**Vorgeschlagene Kernzellen (Pseudocode, füllen wir gemeinsam):**

```python
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
quantiles, mean = pipe.predict_quantiles(
    context=torch.tensor(y_history),
    prediction_length=H,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

---

## 4. Hands-on `2_chronos_handson_students.ipynb` – Gliederung

Aufbau **analog zum Prophet-Hands-on**: Setup, Datengenerierung und Plots sind
vorgegeben (nur ausführen), die Studierenden schreiben den Modell-Teil selbst. Wir
nutzen denselben Spielzeug-Datensatz **„SkyDrive X1 Pro" (SSD-Verkäufe)** wie bei
Prophet – das hält den Vergleich konsistent.

- **Aufgabe 1 – Naiver Zero-Shot-Forecast:** ein Chronos-Bolt-Modell laden und die
  nächsten N Tage vorhersagen, *ganz ohne Konfiguration*. (Gegenstück zum „naiven
  Prophet-Modell ohne Feiertage".)
- **Aufgabe 2 – Modell verbessern:** zwei Varianten zur Auswahl, je nach gewünschter
  Tiefe:
  - **2a (einfach):** Kontextlänge erhöhen / größeres Modell (`bolt-base`) und den
    Effekt auf die Metriken zeigen.
  - **2b (anspruchsvoll, Chronos-2):** den Black Friday als **Kovariate** mitgeben –
    direktes Gegenstück zum „Experten-Modell mit Black Friday" bei Prophet.
- **Showdown:** naiv vs. verbessert, automatischer Vergleichsplot + Diskussionsfragen
  (Wann lohnt sich ein Foundation Model? Was kostet es an Rechenzeit/Erklärbarkeit?).

> **Empfehlung:** 2a als Pflicht, 2b als Bonus – so bleibt die Übung für Anfänger
> machbar, bietet aber Tiefe.

---

## 5. Musterlösung `3_chronos_handson_solution.ipynb`

Identisch zum Hands-on, aber alle TODO-Zellen ausgefüllt + kurze Erläuterungen, warum
welche Wahl. Wie bei Prophet (`3_prophet_handson_solution.ipynb`).

---

## 6. Setup & Abhängigkeiten

Chronos braucht **PyTorch** (haben wir schon in der Prophet-Env) plus das Paket
`chronos-forecasting`:

```bash
pip install chronos-forecasting
```

Die `chronos-tutorial.yml` baut auf der bestehenden Conda-Env auf und ergänzt nur
Chronos. Alle empfohlenen Modelle (Bolt-small/base) laufen auf **CPU**; sie werden beim
ersten `from_pretrained(...)` automatisch von Hugging Face heruntergeladen (einmalig
Internet nötig, danach im Cache).

**Wichtig für die Demo:** Beim ersten Lauf lädt das Modell ein paar hundert MB – das im
Tutorial vorher einmal ausführen, damit es live schnell geht.

---

## 7. Literatur (Vorschlag für `literature/`)

- **Ansari et al. (2024). *Chronos: Learning the Language of Time Series.*** TMLR.
  arXiv:2403.07815 – das Original-Paper.
- **Ansari et al. (2025). *Chronos-2: From Univariate to Universal Forecasting.***
  arXiv:2510.15821 – Kovariaten & multivariate Vorhersage.
- Optional: der AWS-Blogpost zu **Chronos-Bolt** (schnelle Variante) als gut lesbare
  Praxis-Quelle.

---

## 8. Offene Entscheidungen (bitte kurz abnicken)

1. **Modellgröße in der Demo:** Bolt-`small` (schnell) oder Bolt-`base` (genauer)? –
   Vorschlag: `small` zeigen, `base` als „mehr Rechenzeit = besser?"-Vergleich.
2. **Chronos-2-Kovariaten (Schritt 5 / Aufgabe 2b):** mit reinnehmen oder bewusst
   weglassen, um es einfach zu halten? – Vorschlag: in der Demo zeigen, im Hands-on als
   Bonus.
3. **Hands-on-Daten:** SkyDrive-SSD wie bei Prophet (Vergleich) – einverstanden?
4. **PowerPoint:** neue Chronos-Einstiegspräsentation, oder eine gemeinsame Folienreihe
   „Prophet vs. Chronos"? – Das klären wir, wenn die Notebooks stehen.
