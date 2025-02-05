import pandas as pd
import pickle as pk
import json
from tqdm import tqdm

for modelname in ["gtelarge"]:
    col = f"inverse_cluster_frequency_{modelname}"
    creators = ["children", "adults", "AI"]
    for creator in creators:
        df = pd.read_csv(f"../../csvs/{creator}_product.csv")
        df[col] = None

        cluster_infos = {
                            "G": pk.load(open(f"../../saved/cluster_info/cluster_info_{creator}_G_{modelname}.pk", "rb")),
                            "I": pk.load(open(f"../../saved/cluster_info/cluster_info_{creator}_I_{modelname}.pk", "rb")),
                            "R": pk.load(open(f"../../saved/cluster_info/cluster_info_{creator}_R_{modelname}.pk", "rb"))
                        }

        df[col] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows for {creator}"):
            stim = row["stimuli"]
            response_to_cluster, cluster_to_response = cluster_infos[stim]
            den = len(response_to_cluster)
            df.at[index, col] = den/len(cluster_to_response[response_to_cluster[row["gpt_caption"]]])
        df.to_csv(f"../../csvs/{creator}_product.csv", index=False)