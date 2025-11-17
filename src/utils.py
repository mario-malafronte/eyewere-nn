# src/utils.py
import re
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

KEYWORDS = [
    "polarized","photochromic","aviator","wayfarer","wrap","sports","safety","uv400",
    "retro","square","round","mirrored","gradient","metal","plastic","acetate","carbon fiber",
    "cycling","fishing","driving","golf","women","men","unisex","pack"
]

def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int,float)):
        return float(x)
    s = str(x).strip()
    s = s.replace("$","").replace("€","").replace(",","")
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    return float(m[0]) if m else np.nan

def load_dataset(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["title","brand","description","currency","price","stars","reviewsCount"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne nel CSV: {missing}")
    return df

def clean_dataset(df:pd.DataFrame, price_min:float=3, price_max:float=500)->pd.DataFrame:
    out = df.copy()
    out["price"] = out["price"].apply(to_float)
    out["rating"] = out["stars"].apply(to_float).clip(0,5)
    out["reviews"] = out["reviewsCount"].apply(to_float).fillna(0).astype(int)
    out["title"] = out["title"].fillna("")
    out["description"] = out["description"].fillna("")
    out["brand"] = out["brand"].fillna("Other").astype(str)

    out = out[out["price"].between(price_min, price_max)]
    out = out[out["brand"].str.len()>0]
    out = out[out["title"].str.len()>0]
    
    out = out.drop_duplicates(subset=["title","brand","price"], keep="first")
    out.reset_index(drop=True, inplace=True)
    return out

def build_features(df: pd.DataFrame):
    X_num = pd.DataFrame()
    X_num["rating"]  = df["rating"].fillna(df["rating"].median())
    X_num["reviews"] = df["reviews"].fillna(0)

    text_all = (df["title"].astype(str) + " " + df["description"].astype(str)).str.lower()
    for kw in KEYWORDS:
        X_num[f"kw_{kw}"] = text_all.str.contains(rf"\b{re.escape(kw)}\b", regex=True).astype(int)

    X = X_num.astype(float)
    y = df["price"].values.astype(float)

    meta = {
        "continuous_cols": ["rating", "reviews"],
        "binary_cols": [f"kw_{k}" for k in KEYWORDS],
        "feature_names": X.columns.tolist(),
    }
    return X, y, meta


