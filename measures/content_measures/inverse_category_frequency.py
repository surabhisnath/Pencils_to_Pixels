import pandas as pd
import pickle as pk
import json
from tqdm import tqdm

col = "inverse_category_frequency"
for creator in ["children", "adults", "AI"]:
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    j1, j2, j3 = [json.load(open(f"../../saved/frequencies/category_frequencies_{creator}_G.pk", "r")), json.load(open(f"../../saved/frequencies/category_frequencies_{creator}_I.pk", "r")), json.load(open(f"../../saved/frequencies/category_frequencies_{creator}_R.pk", "r"))]
    category_freqs = {
                    "G": j1,
                    "I": j2,
                    "R": j3
                }
    den = {
            "G": sum(list(j1.values())), 
            "I": sum(list(j2.values())),
            "R": sum(list(j3.values()))
        }

    df[col] = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
        stim = row["stimuli"]
        cat = row["category1"].replace("(s)", "")
        df.at[index, col] = den[stim]/category_freqs[stim][cat]
    df.to_csv(f"../../csvs/{creator}_product.csv", index=False)