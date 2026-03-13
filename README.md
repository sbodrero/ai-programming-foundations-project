# NYC Airbnb Listings — Data Science Workflow

**Author:** Sébastien Bodrero
**Course:** Woolf / Udacity MSc in Artificial Intelligence — Module 1 Capstone Exercise
**Date:** March 2026

---

## Project Description

This project applies a full data science workflow to the **NYC Airbnb Open Data 2019** dataset, covering approximately 49,000 listings across the five boroughs of New York City.  The goal is to clean, explore, and visualize the data to surface pricing patterns, listing characteristics, and borough-level trends — establishing a reproducible, well-documented baseline for more advanced modelling later in the programme.

The workflow follows six structured phases: Setup → Ingestion → Cleaning → EDA → Visualizations → Summary.

---

## Dataset

- **Name:** NYC Airbnb Open Data 2019 (`AB_NYC_2019.csv`)
- **Source:** Originally published on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) by Dgomonov; also mirrored on GitHub for reproducibility.
- **Size:** ~49,000 rows × 16 columns
- **Key columns:** `neighbourhood_group`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `availability_365`

The notebook auto-downloads the CSV at runtime if it is not already present in `data/AB_NYC_2019.csv`.

---

## How to Install

```bash
cd p1
python -m venv .venv           # or: python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

**Interactive (recommended):**

```bash
cd p1
source .venv/bin/activate
jupyter lab data_workflow.ipynb
```

**Non-interactive / CI check:**

```bash
cd p1
source .venv/bin/activate
jupyter nbconvert --to notebook --execute data_workflow.ipynb \
    --output data_workflow_executed.ipynb
```

The executed notebook will be saved as `data_workflow_executed.ipynb`.

---

## Bias Reflection

Poor data cleaning can introduce several forms of bias into downstream analyses and models:

1. **Survivorship bias from dropping `reviews_per_month` NaNs.** Listings with no review rate are typically brand-new or never-booked.  Dropping them removes an entire population segment, leaving only "active" listings in the dataset.  Any pricing model trained on this subset will not generalise to new listings.

2. **Outlier inflation of mean statistics.** A handful of luxury listings at $5,000–$10,000/night dramatically inflate borough-level mean prices.  Without explicit outlier treatment, summary statistics misrepresent what a typical traveller would pay, creating a misleading picture of affordability.

3. **Geographic coarseness.** Grouping by borough (5 categories) masks enormous within-borough variation — Tribeca and Harlem are both "Manhattan," but price distributions are very different.  Coarser aggregations can suggest false equity between sub-markets.

4. **Temporal snapshot bias.** The dataset is a single 2019 snapshot.  Seasonal pricing patterns (summer peaks, holiday spikes) are invisible, and the dataset cannot represent post-COVID market shifts.  Models trained on it should not be applied to current pricing decisions without recalibration.

---

## Future Integration Reflections

### How This Workflow Would Change in a Full ML Pipeline

In a production ML setting, several phases of this notebook would be rethought:

- **Feature engineering** would replace many one-off EDA steps: creating interaction terms (`borough × room_type`), encoding categoricals (target encoding or embeddings for `neighbourhood`), and deriving temporal features from `last_review` date.
- **Train / validation / test splitting** must happen *before* any imputation or scaling to prevent data leakage — a discipline that is easy to violate when working exploratively in a notebook.
- **Automated retraining** would require the ingestion cell to pull from a live data source (e.g., an Airbnb scraping API or a data warehouse) rather than a static CSV mirror.

### Neural Network Data Preparation

If price prediction were approached with a neural network (e.g., a tabular model using embeddings):

- **Log-transforming the target** (`log(price + 1)`) is essential given the right-skewed distribution observed in Plot 2; the network's loss surface becomes much smoother.
- **Entity embeddings** for high-cardinality categoricals (`neighbourhood`, `host_id`) replace one-hot encoding and capture latent geographic structure efficiently.
- **Normalising numeric inputs** (min-max or z-score) is required because neural networks are sensitive to feature scale — something tree-based models tolerate but MLPs do not.
- **Missing value strategy** shifts from simple row-dropping to learned imputation layers or mask tokens, so the model itself handles incomplete inputs rather than discarding information.

### Agentic Automation Potential

This workflow is a strong candidate for agentic automation:

- **Data ingestion agent:** Monitor the upstream Kaggle / mirror URL for dataset updates; automatically re-trigger the pipeline when a newer version is published.
- **Cleaning agent:** Apply heuristic rules (e.g., "flag prices > 3σ above borough median") autonomously, log decisions, and surface edge cases to a human reviewer only when confidence is low.
- **EDA summarisation agent:** Generate narrative descriptions of each plot automatically (as a first draft for the summary section), using a language model to translate statistical outputs into plain-English insights.
- **Report generation agent:** Assemble the module_summary report from structured outputs (tables, plots, narrative blocks) without manual copy-paste, keeping the final document in sync with the code at all times.

These agents would collectively reduce the time from "raw data arrives" to "report ready for review" from hours to minutes, while keeping a human in the loop for judgement calls.

---

## References

Adhikari, N. K. J. (2022). *Reproducible Data Science with Python: An Open Learning Resource*. ResearchGate. https://doi.org/10.13140/RG.2.2.22099.04641
