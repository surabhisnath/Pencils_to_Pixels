import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pickle as pk

def get_embeddings(texts, modelname, keys):
    """Extracts Text Embeddings using SentenceTransformer (model: gte-large)
    Args:
        texts (list): List of texts
    Returns:
        dict: Text and corresponding embedding
    """
    if modelname == "stella": 
        model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    elif modelname == "gte":
        model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    elif modelname == "gtelarge":
        model = SentenceTransformer('thenlper/gte-large', trust_remote_code=True)
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)         # normalise embeddings
    return dict(zip(keys, embeddings))

modelname = "stella"

base_shape_captions = {
                        "G": "simple line drawing resembling umbrella, mushroom, jellyfish, hot air balloon, igloo",
                        "I": "simple line drawing resembling door, arch, window, tombstone, padlock",
                        "R": "simple line drawing resembling house, arrow, shield, diamond, kite",
                    }

filename = f"base_embeddings_{modelname}"
stimuli = ["G", "I", "R"]
base_embeddings = get_embeddings(list(base_shape_captions.values()), modelname, stimuli)
pk.dump(base_embeddings, open(f"../../saved/text_embeddings/{filename}.pk", "wb"))

for creator in ["children", "adults", "AI"]:
    filename = f"embeddings_{creator}_{modelname}"
    product = pd.read_csv(f"../../csvs/{creator}_product.csv")
    captions = product["gpt_caption"].tolist()
    embeddings = get_embeddings(captions, modelname, captions)
    pk.dump(embeddings, open(f"../../saved/text_embeddings/{filename}.pk", "wb"))