# Analysis of FIT test non-return in Oxfordshire

This code accompanies the manuscript "Should patient characteristics or the time from test request trigger reminders to return Faecal Immunochemical Tests in primary care?", authored by Andres Tamm, Madison Luick, Carmen Fierro Martinez, Brian Shine, Tim James, James E. East, Catia Nicodemo, Stavros Petrou, and Brian D. Nicholson.

Andres Tamm

2025-12-22

## Installation

Install python packages
```bash
pip install -r requirements.txt
```
... and R packages by running `Rpackages.R`.

## Usage

Code was run in the following order:

1. `datacopy.py` : combines data extracts from the clinical datawarehouse into a single folder
2. `dataprep.py` : prepares the dataset for modelling, such as creating 70-day non-return indicators
3. `models.R` : FIT logistic and generalised additive models in R
4. `plots.py` : create graphs that illustrate model performance and test return probabilities over time
5. `tables.py` : create descriptive statistics tables


