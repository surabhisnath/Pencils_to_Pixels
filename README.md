# Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable) [![python](https://img.shields.io/badge/R-4.1.1-3776AB.svg?style=flat&logo=R&logoColor=white)](https://www.python.org)

This repository contains the data and scripts for the paper [Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI](https://www.arxiv.org/pdf/2502.05999) submitted to [CogSci 2025](https://cognitivesciencesociety.org/cogsci-2025/).
Authors: [Surabhi S Nath](https://surabhisnath.github.io), [Guiomar del Cuvillo y Schroder](https://www.researchgate.net/profile/Guiomar-Andrea-Del-Cuvillo-Y-Schroeder), [Claire Stevenson](https://www.uva.nl/en/profile/s/t/c.e.stevenson/c.e.stevenson.html)

## Abstract

Can we derive computational metrics to quantify visual creativity in drawings across intelligent agents, while accounting for inherent differences in technical skill and style? To answer this, we curate a novel dataset consisting of 1338 drawings by children, adults and AI on a creative drawing task. We characterize two aspects of the drawings -- (1) style and (2) content. For style, we define measures of ink density, ink distribution and number of elements. For content, we use expert-annotated categories to study conceptual diversity, and image and text embeddings to compute distance measures. We compare the style, content and creativity of children, adults and AI drawings and build simple models to predict expert and automated creativity scores. We find significant differences in style and content in the groups -- children's drawings had more components, AI drawings had greater ink density, and adult drawings revealed maximum conceptual diversity. Notably, we highlight a misalignment between creativity judgments obtained through expert and automated ratings and discuss its implications. Through these efforts, our work provides, to the best of our knowledge, the first framework for studying human and artificial creativity beyond the textual modality, and attempts to arrive at the domain-agnostic principles underlying creativity.

![Graphical Abstract](figures/graphical_abstract.pdf)

## Repository Description

1. `data` contains the raw data files of children, adults and AI drawings
2. `stimuli` contains the stimulus shapes used in the study
3. `csvs` contains the all processed product and process information of children, adults and AI
4. `measures` contains scripts for generating style (line thickness, ink density, ink fraction inside mask, number of components, number of lines), content (captions, distance from stim, inverse frequency, 10NN image, 10NN text). Some of them cannot be implemented without accessing text and image embeddings which are too large to share on GitHub. But all the measures are already executed and added to the csv. Evaluation contains the scripts for obtaining automated originality (AudrA, OCS) scores.
5. `scripts/children`, `scripts/adults`, `scripts/AI` contain preprocessing and csv creation scripts respectively for children, adults and AI
6. `scripts/Analysis` contain all analyses, separated into Question 1, Question 2 in `Q1Q2` and Question 3 in `Q3`
7. `figures` contain all figures from the paper

## Setup

We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps:

```bash
git clone https://github.com/surabhisnath/Pencils_to_Pixels.git
cd Pencils_to_Pixels

python3 -m venv .env

# On macOS/Linux
source .env/bin/activate
# On Windows
.env\Scripts\activate

pip install -r requirements.txt
```

## Running the code

To reproduce the results from the paper,

1. Run Jupyter Lab as follows:

```bash
jupyter-lab
```

    and open the file`scripts/Analysis/Q1Q2/Q1Q2.ipynb` and run the file (either in one go or cell by cell).

2. For running `scripts/Analysis/Q3/Q3.R`,

- Ensure you have a working R installation
- Install the required libraries
  - *E.g.*, `install.packages(c("lme4", "ggplot2", "dplyr", "lmerTest"))` from an R console
- Run `Q3.R` as follows:

```R
setwd('/path/to/Pencils_to_Pixels/scripts/Analysis/Q3/')
source("Q3.R")
```

    On running the file, all plots are saved to`plots/` folder and tables are saved to `model_fits/`.

## Citation

If you found this work useful, please consider citing us:

```
@misc{nath2025pencilspixelssystematicstudy,
      title={Pencils to Pixels: A Systematic Study of Creative Drawings across Children, Adults and AI}, 
      author={Surabhi S Nath and Guiomar del Cuvillo y Schröder and Claire E. Stevenson},
      year={2025},
      eprint={2502.05999},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2502.05999}, 
}
```
