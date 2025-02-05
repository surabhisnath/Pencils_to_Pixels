# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
from collections import Counter
import spacy
import pandas as pd
import pickle as pk
import json

creators = ["children", "adults", "AI"]

for creator in creators:
    df = pd.read_csv(f"../../csvs/{creator}_product.csv")
    nlp = spacy.load("en_core_web_sm")
    for stim in ["G", "I", "R"]:
        df_ = df[df["stimuli"] == stim]
        df_["category1"] = df_["category1"].str.replace("(s)", "", regex=False)
        categories = df_["category1"].dropna().tolist()
        category_freq = dict(sorted(dict(Counter(categories)).items(), key=lambda x: -x[1]))
        json.dump(category_freq, open(f"../../saved/frequencies/category_frequencies_{creator}_{stim}.pk", "w"))