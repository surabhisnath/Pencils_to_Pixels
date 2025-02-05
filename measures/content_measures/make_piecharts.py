import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import json
import re


def convert_dict(input_dict):
    output_dict = {}
    for value in input_dict.values():
        match = re.match(r"^(.*) \((\d+)\)$", value)
        if match:
            text, num = match.groups()
            output_dict[text.strip()] = int(num)
    return output_dict


def merge_dicts_add_values(*dicts):
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            merged_dict[key] = merged_dict.get(key, 0) + value
    return merged_dict


def make_piechart(themes, c, creator):
    labels = list(themes.keys())
    sizes = list(themes.values())

    sorted_indices = np.argsort(sizes)[::-1]  # Get sorted indices (descending order)
    labels = [labels[i] for i in sorted_indices]
    sizes = [sizes[i] for i in sorted_indices]

    max_font = 18
    if creator == "adults":
        min_font = 6
        threshold = 25
    else:
        min_font = 8
        threshold = 15
    num_slices = len(sizes)

    if num_slices > threshold:
        large_sizes = np.linspace(
            max_font, max_font, num_slices - threshold
        )  # Keep large sizes
        small_sizes = np.linspace(
            max_font, min_font, threshold
        )  # Gradually decrease for last 15
        font_sizes = np.concatenate((large_sizes, small_sizes))
    else:
        font_sizes = np.linspace(max_font, min_font, num_slices)

    # cmap = cm.get_cmap(cmap, len(sizes))
    # colors = [(*cmap(i / (len(sizes) - 1))[:3], 0.4) for i in range(len(sizes))]
    # Change this to any color
    colors = [c] * len(sizes)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f"" if p >= 2 else "",
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.40,  # Move % inward
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": 0},
    )

    for wedge, label, font_size, size in zip(wedges, labels, font_sizes, sizes):
        theta = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(theta))
        y = np.sin(np.deg2rad(theta))
        rotation_angle = theta if theta > -90 else theta - 180
        ax.text(
            x * 1.2,
            y * 1.2,
            label,
            rotation=rotation_angle,
            ha="center",
            va="center",
            rotation_mode="anchor",
            fontsize=font_size,
        )
        percentage = (size / sum(sizes)) * 100
        if percentage >= 2:  # Show % only if >= 2
            ax.text(
                x * 0.4,
                y * 0.4,
                f"{percentage:.1f}%",
                rotation=rotation_angle,
                ha="center",
                va="center",
                rotation_mode="anchor",
                fontsize=12,
            )

    plt.savefig(f"../../figures/pie_chart_{creator}.png", bbox_inches="tight", dpi=500)


# ------------------------------------CHILDREN----------------------------------------
children_G = convert_dict(
    json.load(
        open(
            "../../saved/cluster_summaries/cluster_to_summary_children_G_gtelarge.json"
        )
    )
)
children_I = convert_dict(
    json.load(
        open(
            "../../saved/cluster_summaries/cluster_to_summary_children_I_gtelarge.json"
        )
    )
)
children_R = convert_dict(
    json.load(
        open(
            "../../saved/cluster_summaries/cluster_to_summary_children_R_gtelarge.json"
        )
    )
)
children_themes = merge_dicts_add_values(children_G, children_I, children_R)
make_piechart(children_themes, "#E0F7EF", "children")

# ------------------------------------ADULTS----------------------------------------
adults_G = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_adults_G_gtelarge.json")
    )
)
adults_I = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_adults_I_gtelarge.json")
    )
)
adults_R = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_adults_R_gtelarge.json")
    )
)
adults_themes = merge_dicts_add_values(adults_G, adults_I, adults_R)
make_piechart(adults_themes, "#FAD5C5", "adults")

# ------------------------------------AI----------------------------------------
AI_G = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_AI_G_gtelarge.json")
    )
)
AI_I = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_AI_I_gtelarge.json")
    )
)
AI_R = convert_dict(
    json.load(
        open("../../saved/cluster_summaries/cluster_to_summary_AI_R_gtelarge.json")
    )
)
AI_themes = merge_dicts_add_values(AI_G, AI_I, AI_R)
make_piechart(AI_themes, "#D9D2E9", "AI")
