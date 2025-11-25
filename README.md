# Balance vs Socially Fair k-means
This repository provides a unified Python pipeline for generating counterfactual explanations and fairness metrics for clustering results obtained from two fairness-aware clustering algorithms:

Balance Method
https://github.com/nicolasjulioflores/fair_algorithms_for_clustering

Socially Fair k-means (Fair-Lloyd)
https://github.com/samirasamadi/SociallyFairKMeans

This repository does not implement these algorithms.
It consumes their outputs to compute alignment, NMI cost, misalignment, counterfactual distances, feature contributions, and explainability plots.

## Functionality Overview

The notebook implements a full analysis pipeline:

1. Label Alignment
Align fair clustering labels to unfair labels using the Hungarian algorithm.

2. Fairness Metrics
   * Total NMI cost
   * Per-group NMI cost
   * Total misalignment
   * Per-group misalignment

3. For every misaligned instance, we generate a counterfactual (CF) that minimally alters the factual point so that it transitions into its aligned fair cluster.
   This involves:
    * Determining the smallest feature change required for the instance to switch clusters under the fair model.
    * Measuring the distance of the counterfactual to the separating hyperplane (or decision boundary) to quantify how “close” the change is.
    * Storing factual–counterfactual pairs for every method, distance group, seed, and value of k, enabling systematic comparison across clustering and fairness settings.

4. Feature Contribution Analysis
Compute absolute change per feature,
Aggregate across misaligned points,
Bar plots of average contribution,

5. Explainability Plots
Strip plots showing CF change per feature
Colored by factual feature values
Created separately for Balance and Social-Fair
Generated per sensitive group

6. Unified Comparison Plots
NMI cost per group (all methods in one plot)
Counterfactual distance per group (all methods in one plot)

## How to run
1. Run the Balance method repository and generate:
    output/
    unfair_centers_<dataset>/
    fair_centers_<dataset>/

2. Run SociallyFairKMeans (MATLAB) and generate:
    cost_seeds/full_results_seed_*_k_*.mat
3. Open the notebook:
    fair_counterfactuals.ipynb
4. Set the dataset:
   DATASET_NAME = "adult"
5. Run all cells.
# Soft Fair Clustering Framework

This repository provides a research framework for **soft fair clustering** in tabular data.  It implements a full pipeline that balances protected subgroups, applies Gaussian Mixture Models (GMMs) with fairlet decompositions, aligns clusters across fair and unfair models, and quantifies fairness through misalignment, NMI costs, and counterfactual distances.

## Overview

> **Important note on the GMM framework**: This code builds directly upon the *Gaussian mixture model (GMM) fairlet decomposition* introduced in the paper *“Fair Soft Clustering”* by Rune D. Kjærsgaard **et al.** (AISTATS 2024).  The accompanying implementation, available at [`RuneDK93/fair-soft-clustering`](https://github.com/RuneDK93/fair-soft-clustering), provides the core algorithms for fairlet-based GMM clustering.  Our framework adapts and extends this GMM fairlet decomposition for the soft fairness experiments described below.  See the *Background and Credits* section for details and citation information.

Clustering algorithms such as k‑means or Gaussian mixtures often yield cluster assignments that disproportionately disadvantage minority groups.  This framework addresses this issue via **soft fair clustering**, blending traditional clustering objectives with fairness constraints.  The pipeline features:

- **Dataset registry** with preconfigured settings for several common datasets (Adult Income, Student Performance, Bank Marketing, Credit Default).  Each configuration specifies the CSV file, selected features, protected attribute, and a dataset prefix used for saved outputs.
- **Preprocessing utilities** to drop missing values, encode the protected attribute as \(\{0,1\}\), balance the dataset to a 1:2 minority/majority ratio, and scale continuous features.
- **Fairlet decomposition** using Maximum Cardinality Fairlets (MCF) to produce fair cluster representatives before fitting a GMM.
- **Evaluation metrics** that include traditional clustering cost, fairness balance measures, and per‑group NMI (Normalized Mutual Information) to quantify alignment between fair and unfair assignments.
- **Alignment and misaligned analysis** to match fair clusters to unfair ones and identify misassigned points on a per‑group basis.
- **Counterfactual explainability** to compute counterfactual examples for misaligned points and measure feature‑level contributions to the counterfactual distance.

This framework is intended for researchers exploring fairness in clustering.  It provides scripts to run experiments on multiple values of \(k\), compare fair and unfair models, visualise alignment and fairness metrics, and analyse counterfactual explanations.

## Features

- **Multiple datasets:** Supports `adult`, `student`, `bank`, and `credit` data sets (customise or add new ones via `dataset_config`).
- **Configurable experiment:** The top of the main script includes a configuration section where you specify the dataset, the range of clusters \(K\), random seeds, covariance type, (\(p,q\))-fairlet parameters, and output directories.
- **Fairlet decomposition:** Uses `MCFFairletDecomposition` to enforce group balance before clustering, producing a *soft fair clustering* solution.
- **Evaluation:** Computes cluster costs, fairness balances, likelihood, and misalignment metrics.  Generates plots for overall and per‑group NMI cost, misaligned counts, and soft balance comparison with the baseline.
- **Counterfactual analysis:** Generates counterfactual data points for misaligned observations and calculates the distance distribution between original and counterfactual instances.  Summarises feature contributions to the counterfactual distance.

## Data Setup

1. **Prepare CSV files** corresponding to your chosen dataset.  For each dataset, the configuration in `dataset_config` lists the expected columns and the name of the protected attribute.  Place these files in the working directory or adjust the `csv_path` accordingly.
2. **Choose your dataset** by setting the `DATASET` variable at the top of the script to one of `adult`, `student`, `bank`, or `credit`.
3. **Configure experiment parameters:** Adjust `K_MIN`, `K_MAX`, `SEEDS`, `COV_TYPE`, `FAIR_PQ`, and other parameters as needed.  `FAIR_PQ` sets the (\(p,q\))‑fairlet decomposition ratio; by default it is `(1, 2)`, meaning each fairlet contains one protected‑minority point for every two majority points.

## Usage

### Installation

1. Clone this repository and navigate to its root.
2. Ensure that Python 3.8+ is installed.  Install dependencies with:
   ```bash
   pip install -r requirements.txt
