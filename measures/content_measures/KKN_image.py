import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import pickle as pk
import pandas as pd

def find_mean_cosine_distances(embeddings, top_k=10):
    distances = cosine_distances(embeddings, embeddings)
    mean_distances = []
    
    for i in range(len(embeddings)):
        nearest_distances = np.sort(distances[i])[1:top_k+1]
        mean_distances.append(np.mean(nearest_distances))
    print(mean_distances)
    return mean_distances


modelnames = ["dino", "clip"]

for modelname in modelnames:
    col = f"image_10NN_{modelname}"
    creators = ["children", "adults", "AI"]
    for creator in creators:
        embeddings = pk.load(open(f"../../saved/image_embeddings/embeddings_{creator}_{modelname}.pk", "rb"))
        df = pd.read_csv(f"../../csvs/{creator}_product.csv")
        df[col] = None
        for stim in ["G", "I", "R"]:
            images = df[df["stimuli"] == stim]["filepath"].tolist()
            df.loc[df["stimuli"] == stim, col] = find_mean_cosine_distances([embeddings[x] for x in images])
        df.to_csv(f"../../csvs/{creator}_product.csv", index=False)