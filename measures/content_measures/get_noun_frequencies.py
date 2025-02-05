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
    
    captions = df["gpt_caption"].dropna().tolist() 
    texts = " ".join(captions)
    doc = nlp(texts)
    nouns = [token.lemma_.lower() for token in doc if token.pos_ == "NOUN"]
    noun_freq = dict(sorted(dict(Counter(nouns)).items(), key=lambda x: -x[1]))
    json.dump(noun_freq, open(f"../../saved/frequencies/noun_frequencies_gptcaption_{creator}.json", "w"))

    categories = df["category1"].dropna().tolist() + df["category2"].dropna().tolist() + df["category3"].dropna().tolist()
    texts = " ".join(categories)
    doc = nlp(texts)
    nouns = [token.lemma_.lower().replace("(", "") for token in doc if token.pos_ == "NOUN"]
    noun_freq = dict(sorted(dict(Counter(nouns)).items(), key=lambda x: -x[1]))
    json.dump(noun_freq, open(f"../../saved/frequencies/noun_frequencies_categories_{creator}.json", "w"))