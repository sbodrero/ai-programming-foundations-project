# Module Summary Report — APA 7 Template
## NYC Airbnb Listings: A Data Science Workflow

---

**Title:** NYC Airbnb Listings: A Complete Data Science Workflow for Pricing Analysis

**Author:** Sébastien Bodrero

**Institutional Affiliation:** Woolf University / Udacity MSc in Artificial Intelligence

**Course:** AI Mastery — Module 1 Capstone Exercise

**Date:** March 12, 2026

---

## Overview

This report documents the design, implementation, and findings of a complete data science workflow applied to the New York City Airbnb Open Data 2019 dataset. The workflow follows six structured phases — setup, ingestion, cleaning, exploratory data analysis (EDA), visualisation, and summary interpretation — and is implemented as a fully executable Jupyter Notebook (`data_workflow.ipynb`). The primary goal is to surface pricing patterns and listing characteristics across NYC's five boroughs, while establishing a reproducible baseline suitable for future machine learning integration.

The approach taken here is grounded in reproducible data science practices. As Adhikari (2022) notes, a reproducible workflow "ensures that others can verify, build on, and extend [your] analysis," which is particularly important in academic and professional settings where transparency of method is as valuable as the findings themselves.

---

## Dataset Description

The **NYC Airbnb Open Data 2019** dataset contains 48,895 listings scraped from the Airbnb platform and made publicly available on Kaggle (Dgomonov, 2019). The dataset includes 16 columns covering:

- **Host information:** `host_id`, `host_name`, `calculated_host_listings_count`
- **Location:** `neighbourhood_group` (borough), `neighbourhood`, `latitude`, `longitude`
- **Listing type:** `room_type` (Entire home/apt, Private room, Shared room)
- **Pricing and availability:** `price`, `minimum_nights`, `availability_365`
- **Review activity:** `number_of_reviews`, `last_review`, `reviews_per_month`

The dataset represents a point-in-time snapshot of the NYC Airbnb market in 2019, prior to the disruptions caused by the COVID-19 pandemic. This temporal limitation is an important consideration when interpreting findings.

**Source:** Originally published at https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data. A public GitHub mirror at `https://raw.githubusercontent.com/erkansirin78/datasets/master/AB_NYC_2019.csv` is used for automated download within the notebook.

---

## Workflow Description

The workflow was implemented in six sequential phases:

### 1. Setup

Standard data science libraries were imported: `numpy` and `pandas` for data manipulation; `matplotlib` and `seaborn` for visualisation. A consistent visual theme was applied via `sns.set_theme()` to ensure all plots share a coherent style.

### 2. Ingestion

The dataset is downloaded programmatically using Python's `urllib.request` module if not already present locally. Once available, it is loaded via `pd.read_csv()` and inspected with `.head()` and `.info()` to understand shape, column types, and the presence of null values before any transformation is applied.

### 3. Cleaning

Two student-defined cleaning functions were implemented:

- **`drop_missing_reviews(df)`:** Removes all rows where `reviews_per_month` is NaN. Listings with no review rate represent new or unlisted properties that have never been booked, making them structurally different from active listings. Dropping them (rather than imputing zero) avoids introducing a false signal in price–review correlation analyses.

- **`fix_column_types(df)`:** Coerces `price` and `minimum_nights` to numeric types using `pd.to_numeric(..., errors='coerce')`, stripping any stray currency symbols before conversion. Rows that cannot be coerced are dropped. This ensures all downstream arithmetic operations produce correct results.

Additionally, zero-price listings (likely data-entry errors) were removed. After cleaning, 38,843 rows remained from the original 48,895.

### 4. Exploratory Data Analysis (EDA)

A single EDA function, **`explore_by_neighbourhood_group(df)`**, computed per-borough aggregate statistics: listing count, mean and median price, mean availability, and mean review rate. Results were sorted by mean price descending to immediately surface the pricing hierarchy across boroughs.

A supplementary room-type breakdown computed the same aggregations grouped by `room_type` to compare pricing across Entire home/apt, Private room, and Shared room categories.

### 5. Visualisations

Three plots were produced, each with a title and labelled axes:

1. **Bar chart — Average nightly price by borough:** Displays the five boroughs ranked by mean price, with values annotated on each bar.
2. **Histogram — Price distribution (log scale, < $500):** Shows the right-skewed, approximately log-normal distribution of nightly prices, with median and mean marked as vertical reference lines.
3. **Heatmap — Correlation matrix:** Shows pairwise Pearson correlations among the six numeric features, using a diverging colour map centred at zero.

Each visualisation is accompanied by an interpretive Markdown cell in the notebook.

### 6. Summary

The final section synthesises findings, identifies patterns and surprises, articulates assumptions, and notes limitations of the analysis. It also includes the full APA-format reference for the primary cited source.

---

## Key Decisions and Assumptions

| Decision | Justification |
|---|---|
| Drop rows with missing `reviews_per_month` | New/never-booked listings are a structurally distinct population; imputing zero would falsely suggest low engagement |
| Coerce rather than error on type mismatches | Produces a usable dataset even if a handful of rows contain formatting artifacts |
| Filter price < $500 for histogram only | Extreme outliers compress the visible distribution; the full dataset is retained for other analyses |
| Log-scale x-axis on price histogram | The distribution spans two orders of magnitude; log scale makes the modal region readable |
| Aggregate at borough level (not neighbourhood) | 5 boroughs provide stable group sizes; 221 neighbourhoods would produce unreliable estimates for small groups |

---

## Results and Interpretation

**Pricing hierarchy:** Manhattan commands the highest average nightly price (~$185), followed by Brooklyn (~$124), Queens (~$100), Staten Island (~$115), and the Bronx (~$88). The Manhattan premium reflects its status as the city's primary commercial and tourist hub.

**Price distribution:** The nightly price distribution is approximately log-normal with a mode near $75. The mean ($151) is substantially higher than the median ($101), confirming the right-skewed nature of the data driven by a minority of high-priced luxury listings.

**Room type premium:** Entire home/apartment listings have a median price roughly twice that of private rooms, and approximately three times that of shared rooms. This confirms room type as a second major pricing driver after location.

**Review–price relationship:** Higher-priced listings attract fewer reviews on average, consistent with lower booking frequency. This negative correlation is weak (r ≈ −0.05) but consistent in direction, suggesting that review rate could serve as a proxy for demand in predictive models.

**Host multi-listing effect:** `calculated_host_listings_count` shows a modest positive correlation with price (r ≈ 0.06), suggesting that professional hosts managing multiple properties may price slightly higher, possibly due to better amenities or marketing.

---

## Responsible Practice: Bias and Data Quality

Several data-quality and bias considerations were identified:

1. **Survivorship bias.** Dropping ~10,000 rows with no `reviews_per_month` removes new listings entirely. Any model trained on the resulting dataset will not generalise to unlaunched or newly listed properties.

2. **Outlier-driven mean distortion.** A small number of extreme-price listings (> $1,000/night) significantly inflate mean statistics. Median price is a more robust summary for this dataset.

3. **Geographic aggregation bias.** The borough-level aggregation masks substantial within-borough heterogeneity. Tribeca and East Harlem are both "Manhattan," but represent very different market segments.

4. **Temporal snapshot limitation.** The 2019 data cannot capture post-COVID supply contraction, regulatory changes (e.g., Local Law 18, NYC's 2023 short-term rental regulations), or shifting demand patterns. Findings should not be applied to current market conditions without re-validation.

5. **Reported vs. actual prices.** The `price` column reflects list price, not booking price. Dynamic pricing, discounts, and seasonal adjustments mean the dataset may understate real transaction prices for popular periods.

---

## Reproducibility

The notebook is designed to be fully reproducible by any user with Python 3.10+ and internet access:

- **Automated download:** The ingestion cell checks for the local CSV and downloads it automatically if absent.
- **Pinned dependencies:** All package versions are recorded in `requirements.txt` (generated via `pip freeze`).
- **Top-to-bottom execution:** The notebook executes without errors from a clean kernel using `jupyter nbconvert --to notebook --execute`.
- **Version control:** Development occurred on the `dev` branch of the GitHub repository, with `main` serving as the stable reference branch. Multiple commits document the incremental development of each section.

Adhikari (2022) identifies dependency pinning and automated execution checks as foundational requirements for reproducibility in data science projects — both are implemented here.

---

## References

Adhikari, N. K. J. (2022). *Reproducible Data Science with Python: An Open Learning Resource*. ResearchGate. https://doi.org/10.13140/RG.2.2.22099.04641

Dgomonov. (2019). *New York City Airbnb Open Data* [Dataset]. Kaggle. https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
