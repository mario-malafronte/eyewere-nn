# src/train_tf.py
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from .utils import load_dataset, clean_dataset, build_features

def split_train_val_test(n, test_size=0.15, val_size=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Dataset troppo piccolo rispetto a test/val size.")
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def standardize_numeric_frames(X_train, X_val, X_test, numeric_cols):
    mu = X_train[numeric_cols].mean(axis=0)
    sigma = X_train[numeric_cols].std(axis=0).replace(0, 1.0)
    X_train_sc = X_train.copy(); X_val_sc = X_val.copy(); X_test_sc = X_test.copy()
    X_train_sc[numeric_cols] = (X_train[numeric_cols] - mu) / sigma
    X_val_sc[numeric_cols]   = (X_val[numeric_cols] - mu) / sigma
    X_test_sc[numeric_cols]  = (X_test[numeric_cols] - mu) / sigma
    return X_train_sc, X_val_sc, X_test_sc, {
        "numeric_cols": list(numeric_cols),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }

def build_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def metrics_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/eyewear.csv", help="Path CSV input")
    ap.add_argument("--outdir", type=str, default="outputs", help="Directory per output/plot/predizioni")
    ap.add_argument("--models_dir", type=str, default="models", help="Directory per il modello salvato")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_dataset(args.data)
    df = clean_dataset(df)

    X_df, y, meta = build_features(df)
    n = len(X_df)
    train_idx, val_idx, test_idx = split_train_val_test(n, args.test_size, args.val_size, args.seed)
    X_train = X_df.iloc[train_idx].copy(); y_train = y[train_idx]
    X_val   = X_df.iloc[val_idx].copy();   y_val   = y[val_idx]
    X_test  = X_df.iloc[test_idx].copy();  y_test  = y[test_idx]

    continuous_cols = meta["continuous_cols"]
    X_train_sc, X_val_sc, X_test_sc, norm_meta = standardize_numeric_frames(X_train, X_val, X_test, continuous_cols)

    Xtr = X_train_sc.values.astype("float32"); ytr = y_train.astype("float32")
    Xva = X_val_sc.values.astype("float32");   yva = y_val.astype("float32")
    Xte = X_test_sc.values.astype("float32");  yte = y_test.astype("float32")

    model = build_model(input_dim=Xtr.shape[1])
    es = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es],
        verbose=0
    )

    y_val_pred  = model.predict(Xva, verbose=0).reshape(-1)
    y_test_pred = model.predict(Xte, verbose=0).reshape(-1)
    val_m = metrics_np(yva, y_val_pred)
    test_m = metrics_np(yte, y_test_pred)

    model_path = os.path.join(args.models_dir, "tf_model.keras")
    model.save(model_path)
    with open(os.path.join(args.models_dir, "tf_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "continuous_cols": norm_meta["numeric_cols"], 
            "mu": norm_meta["mu"],
            "sigma": norm_meta["sigma"],
            "feature_names": X_df.columns.tolist(),
            "extra_meta": meta
        }, f, indent=2, ensure_ascii=False)


    pred_df = pd.DataFrame({
        "title": df.loc[X_test.index, "title"].values,
        "brand": df.loc[X_test.index, "brand"].values,
        "price_actual": y_test,
        "price_pred": y_test_pred,
    })
    pred_df["abs_error"] = (pred_df["price_actual"] - pred_df["price_pred"]).abs()
    pred_df.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    metrics_obj = {
        "rows_after_cleaning": int(len(df)),
        "n_features": int(X_df.shape[1]),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "val_metrics": val_m,
        "test_metrics": test_m,
        "model_info": {"backend": "keras", "path": model_path},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2, ensure_ascii=False)

    plt.figure()
    plt.scatter(range(len(y_test)), y_test, label="actual")
    plt.scatter(range(len(y_test_pred)), y_test_pred, label="pred")
    plt.title("Prezzo — Test set (TensorFlow)")
    plt.legend()
    plt.savefig(os.path.join(args.outdir, "price_fit.png"), dpi=160, bbox_inches="tight")

    print(json.dumps(metrics_obj, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
