# Counterfactual-based-fair-clustering-cost

This repository provides a unified Python pipeline for evaluating fairness in clustering and generating counterfactual explanations using the outputs of multiple fair clustering algorithms.

The repository does not implement these clustering algorithms.Instead, it consumes their outputs and produces:

-Label alignment (Hungarian matching)
-Group-based fairness metrics
-Misalignment and NMI cost analysis
-Counterfactual explanations for misaligned points
-Feature-level contribution analysis
-Explainability strip plots
-Unified comparison plots across fairness methods

This pipeline works with the outputs of the following algorithms:

**Balance Method**
https://github.com/nicolasjulioflores/fair_algorithms_for_clustering

**Socially Fair k-means (Fair-Lloyd)**
https://github.com/samirasamadi/SociallyFairKMeans

**Fair-Soft-Clustering**
This framework also supports outputs from soft fair clustering (GMM + fairlets) as introduced in:
https://github.com/RuneDK93/fair-soft-clustering

This repository is an analysis & explainability toolkit for fair clustering.
It acts as a post-processing pipeline applied after running Balance, Fair-Lloyd, UniFair, or Soft-Balance/GMM.
## Functionality Overview

The notebook implements a full analysis pipeline:

1. Label Alignment
 * Align fair clustering labels to unfair labels using the Hungarian algorithm.

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
 * This module quantifies how individual features contribute to counterfactual changes.
 * For every misaligned instance, we compute the absolute counterfactual change per feature
 * These changes are then aggregated across all misaligned points to identify which features require the largest adjustments.

5. Explainability Plots
 * Visualizations that highlight how features change when generating counterfactuals.
 * Uses strip plots to show the distribution of counterfactual adjustments for each feature.
 * Each point represents a misaligned instance; color encodes the original factual feature value.
 * Plots are generated separately for Balance Fairness and Social Fairness, enabling comparison.
 * Outputs are also produced per sensitive group (e.g., male/female), showing fairness behavior across subpopulations.

6. Unified Comparison Plots
 * Combined summaries comparing all fairness method
 * **NMI cost per group:**
  - One plot showing Normalized Mutual Information loss for each sensitive group across all clustering methods.
 * **Counterfactual distance per group:**
  - One plot showing average CF distances required to fix misalignments for each group.
These unified plots make it easy to compare Balance, Social Fairness, Fair-Lloyd, UniFair, or any additional methods in a single visualization.

## How to run
1. Run the Balance method repository and generate:
    ```bash
    output/
    unfair_centers_<dataset>/
    fair_centers_<dataset>/

2. Run SociallyFairKMeans (MATLAB) and generate:
    ```bash
    cost_seeds/full_results_seed_*_k_*.mat
3. Open the notebook:
    ```bash
    fair_counterfactuals.ipynb
4. Set the dataset:
   ```bash
   DATASET_NAME = "adult"

5. Run all cells.

# Soft Fair Clustering Framework
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


### Installation

1. Clone this repository and navigate to its root.
2. Ensure that Python 3.8+ is installed.  Install dependencies with:
   ```bash
   pip install -r requirements.txt
   
### Acknowledgments
The research project is implemented in the framework of H.F.R.I. call ``Basic research Financing (Horizontal support of all Sciences)'' under the National Recovery and Resilience Plan ``Greece 2.0'' funded by the European Union - NextGenerationEU (H.F.R.I. ProjectNumber: 15940).
