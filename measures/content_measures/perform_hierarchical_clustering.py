import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="Verdana")
# Remove the gray background grid
sns.set_style("white")
import warnings
warnings.simplefilter(action='ignore')
from wordcloud import WordCloud
import re

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

import pickle as pk
from collections import defaultdict
import os
import together
together.api_key = os.environ.get("TOGETHER_API_KEY")
from tqdm import tqdm

rcParams["text.usetex"] = True
rcParams["font.family"] = "sans-serif"
rcParams["grid.linestyle"] = ":"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.fontsize"] = 13
rcParams["axes.labelsize"] = 18
rcParams["axes.titlesize"] = 20
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15

def get_clusters(embeddings, texts, cluster_assignment, printclusters):
    """Helper function for perform_hierarchicalClustering()
    Args:
        embeddings (list): List of embeddings
        texts (list): List of texts
        cluster_assignment (list): cluster assignment for each text/embedding
        printcluster (bool): True will print all texts in each cluster
    Returns:
        response_to_cluster dict: text to cluster number
        cluster_to_response dict: cluster number to list of texts in that cluster
        num_clusters int: number of clusters
        min_similarities dict: cluster to min text similiarity
    """

    cluster_to_response = defaultdict(list)                                 # make cluster_to_response
    cluster_to_embeddings = defaultdict(list)                               # make cluster_to_embeddings (used for min sim)
    for ind, cluster_num in enumerate(cluster_assignment):
        cluster_to_response[cluster_num].append(texts[ind])
        cluster_to_embeddings[cluster_num].append(embeddings[ind])
    
    response_to_cluster = dict(zip(texts, cluster_assignment))              # make response_to_cluster 
    
    num_clusters = len(np.unique(cluster_assignment))                       # find num clusters

    min_similarities = {}                                                   # find min paiwise text similarity in each cluster
    for cl in cluster_to_embeddings:
        stacked = np.array(cluster_to_embeddings[cl])
        sim = stacked @ stacked.T
        np.fill_diagonal(sim, np.inf)
        min_sim = np.min(sim)
        min_similarities[cl] = min_sim
    
    if printclusters:                                                       # print clusters if True
        for cluster_num, responses in cluster_to_response.items():
            print(f"Cluster {cluster_num}")
            print(responses, end="\n\n")

    return response_to_cluster, cluster_to_response, num_clusters, min_similarities

def perform_hierarchicalClustering(embeddings, texts, ax, cut_off_distance, printclusters):
    """Performs hierarchical clustering
    Args:
        embeddings (list): List of embeddings
        texts (list): List of texts
        ax: axes for plotting
        cut_off_distance (float): distance threshold for hierarchical clustering
        printcluster (bool): True will print all texts in each cluster
    Calls:
        get_clusters()
    """
    linked = linkage(embeddings, 'ward')                                                                                                    # 'ward' distance for measuring distance between clusters
    dendrogram(linked, orientation='top', labels=texts, distance_sort='descending', show_leaf_counts=False, no_labels=True, ax=ax[0])       # Make dendogram
    
    # Elbow plot -- Plot mean minsim/number of clusters as a fn of cut-off distance
    mean_minsemsim = []
    num_clusters = []
    for cod in np.linspace(1, 8, 30):
        cluster_assignment = fcluster(linked, t=cod, criterion='distance')
        _, _, _, minsims = get_clusters(embeddings, texts, cluster_assignment, False)
        mean_minsemsim.append(np.mean(list(minsims.values())))
        num_clusters.append(len(np.unique(cluster_assignment)))
    ax[1].plot(np.linspace(1, 8, 30), mean_minsemsim)
    ax[1].set_xlabel("Cut-off distance"); ax[1].set_ylabel("Mean cluster semantic similarity")
    ax[2].plot(np.linspace(1, 8, 30), num_clusters)
    ax[2].set_xlabel("Cut-off distance"); ax[2].set_ylabel("Number of clusters")

    cluster_assignment = fcluster(linked, t=cut_off_distance, criterion='distance')         # assign clusters using the decided cut_off_distance
    return get_clusters(embeddings, texts, cluster_assignment, printclusters)

def together_call(prompt: str) -> str:
    """
    Calls TOGETHER API with input prompt.
    """
    output = together.Complete.create(
        prompt=prompt,
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0,
        max_tokens=512,
        stop=[")", ")```"],
    )
    text_output = output["choices"][0]["text"].replace("assistant", "").replace("\n", "").replace("```", "")
    match = re.split(r"\)+", text_output, maxsplit=1)
    if match:
        return match[0]
    return text_output

def summarise_cluster(cluster_texts):
    text = ",".join(cluster_texts)
    # prompt = f"Summarise the sentences with a single theme phrase. Also output the mean similarity of the sentences on a scale of 1-5. Output format: (theme comma similarity) in round brackets (). Do not output any other words or text.\nSentences: [{text}]."
    prompt = f"Summarise the sentences with a single content theme phrase. Output format: (theme). Do not output any other words or text.\nSentences: [{text}]. Theme: "
    
    try:
        answer = together_call(prompt)
    except Exception as e:
        print("Error:", e)
        answer = "NA"
    return answer.replace("(", "")

def get_cluster_summaries(cl2resp):
    summaries = {}
    for cl in tqdm(cl2resp):
        out = ""
        while out == "":
            out = summarise_cluster(cl2resp[cl])
        summaries[str(cl)] = out + f" ({len(cl2resp[cl])})"
    return summaries


modelname = "gtelarge"
printclusters = True
summarise = True

for creator in ["children", "adults", "AI"]:
    product = pd.read_csv(f"../../csvs/{creator}_product.csv")
    filename = f"embeddings_{creator}_{modelname}"
    embeddings = pk.load(open(f"../../saved/text_embeddings/{filename}.pk", "rb"))
    for stim in ["G", "I", "R"]:
        print(f"--------------------------------{creator} {stim}--------------------------------")

        captions = product[product["stimuli"] == stim]["gpt_caption"].tolist()
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))
        response_to_cluster, cluster_to_response, num_clusters, minsim = perform_hierarchicalClustering([embeddings[x] for x in captions], captions, ax, 0.75, printclusters)
        filename = f"cluster_info_{creator}_{stim}_{modelname}"
        pk.dump([response_to_cluster, cluster_to_response], open(f"../../saved/cluster_info/{filename}.pk", "wb"))
        print("Num clusters =", num_clusters)
        print(np.min(list(minsim.values())), np.mean(list(minsim.values())))

        if summarise:
            filename = f"cluster_to_summary_{creator}_{stim}_{modelname}"
            cluster_summaries = get_cluster_summaries(cluster_to_response)
            json.dump(cluster_summaries, open(f"../../saved/cluster_summaries/{filename}.json", "w"), sort_keys=True, indent=4)
            print("Cluster summaries:")
            print(cluster_summaries)