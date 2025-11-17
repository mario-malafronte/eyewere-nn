# Eyewear Price ML – **Solo TensorFlow/Keras**

> Progetto semplice che stima il prezzo di un prodotto del mercato occhiali partendo da: **titolo, brand, descrizione, stelle, numero recensioni** (il prezzo del CSV viene usato solo per imparare).

- Script di training: `src/train_tf.py`
- Utilità per pulizia e feature: `src/utils.py`

---

## Requisiti

- **Python** consigliato: 3.10 o 3.11 (TensorFlow è più stabile su queste versioni).
- **Dipendenze**: `numpy`, `pandas`, `matplotlib`, **`tensorflow`**.

Setup rapido:
```bash
python -m venv .venv
.venv\Scripts\activate


pip install -U pip
pip install -r requirements.txt
```

---

## Dataset richiesto (CSV)

Percorso tipico: `data/eyewear.csv` (puoi cambiarlo con `--data`).  
**Colonne obbligatorie (nomi esatti):**
```
title, brand, description, currency, price, stars, reviewsCount
```
Note importanti:
- Vengono tenuti solo i prezzi **tra 3 e 500** euro/dollari (configurabile in `clean_dataset`).
- Le righe duplicate (stesso `title,brand,price`) vengono rimosse.
- Se manca una colonna obbligatoria, il training **si ferma** con errore.

---

## Come lanciare il training

```bash
# Esempio Windows
python -m src.train_tf --data data\eyewear.csv --outdir outputs

```

**Opzioni utili:**
- `--test_size` (default `0.15`) quota test
- `--val_size`  (default `0.15`) quota validation
- `--epochs` (default `300`) epoche massime
- `--batch_size` (default `128`)
- `--models_dir` (default `models`) dove salvare il modello
- `--seed` (default `42`) riproducibilità


---

## Cosa fa davvero (in parole semplici)

1) **Pulizia** (`clean_dataset` in `utils.py`): ripulisce prezzi/valori mancanti, filtra range 3–500, rimuove duplicati.  
2) **Feature** (`build_features`): trasforma testo in numeri:
   - **Numeriche**:  
     `rating` (stelle), `reviews` (n. recensioni), `lens_mm` (se trova “XX mm” nel testo), e tante **parole chiave** binarie (0/1) come `polarized`, `photochromic`, `aviator`, `sports`, `uv400`, `acetate`, ecc. (lista `KEYWORDS` in `utils.py`).
   - **Brand**: one‑hot dei **top 50** brand (ognuno diventa una colonna `brand_<nome>`), gli altri confluiscono in `brand_Other`.
3) **Split**: separa **train/val/test** in modo riproducibile (seed).  
4) **Standardizzazione**: scala **solo** le colonne numeriche con **media/deviazione del train** (niente leakage).  
5) **Rete neurale (Keras)**: 256→128→64 neuroni (ReLU) + Dropout, ottimizzatore **Adam**, perdita **MSE**, metrica **MAE**, **early stopping** su validation.  
6) **Valutazione**: calcola **MAE**, **RMSE**, **R2** su validation e test.  
7) **Salvataggi**: modello `.keras`, meta `.json`, predizioni e grafico.

---

## Output generati

- `models/tf_model.keras` → il modello addestrato (Keras).  
- `models/tf_meta.json` → parametri per standardizzazione e lista feature (serve per riusare il modello).  
- `outputs/metrics.json` → riepilogo:
  ```json
  {
    "rows_after_cleaning": 92,
    "n_features": 27,
    "train_size": 64,
    "val_size": 14,
    "test_size": 14,
    "val_metrics": {
      "MAE": 21.827062606811523,
      "RMSE": 43.13808822631836,
      "R2": 0.11832556274033657
    },
    "test_metrics": {
      "MAE": 4.516445159912109,
      "RMSE": 5.903462886810303,
      "R2": 0.21670430309504507
    },
    "model_info": {
      "backend": "keras",
      "path": "models\\tf_model.keras"
    }
  }
  ```
- `outputs/predictions.csv` → confronto riga per riga: `title, brand, price_actual, price_pred, abs_error`.  
- `outputs/price_fit.png` → scatter “prezzi veri vs. predetti” sul test.

**Interpretazione rapida metriche:**
- **MAE**: errore medio in **euro**. Es: 4.51 ⇒ sbaglia in media di ~€4,51.  
- **RMSE**: penalizza di più gli errori grandi.  
- **R2**: quanto meglio della “media costante” (0 = pari alla media, 1 = perfetto).

---

## Perché può funzionare (e perché no)

- Il modello **impara correlazioni** dai dati: parole come `polarized`, tante recensioni, ecc. spesso correlano con prezzi più alti.  


**Limiti principali:**
- **Dataset piccolo** ⇒ risultati instabili, R² basso è normale.  
- **Parole chiave** manuali ⇒ sinonimi/altre lingue possono sfuggire.    
- **Filtro prezzi 3–500**: modificalo se hai premium fuori range.  


---

