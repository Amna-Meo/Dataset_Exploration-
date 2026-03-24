# Iris Dataset Exploration

A Python project for exploring and visualizing the classic Iris dataset using pandas, matplotlib, and seaborn.

## Overview

This project performs comprehensive exploratory data analysis (EDA) on the Iris dataset, including statistical summaries and multiple visualizations to understand the relationships between features and species.

## Features

- Dataset loading and preprocessing
- Statistical analysis (shape, summary statistics, data types)
- Multiple visualization types:
  - Scatter plots (sepal and petal dimensions)
  - Histograms (feature distributions)
  - Box plots (species comparisons)
  - Correlation heatmap
- High-resolution output images (300 DPI)

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

Run the exploration script:
```bash
python iris_exploration.py
```

The script will:
- Print dataset information and statistics to the console
- Generate `iris_exploration.png` with 9 different visualizations
- Generate `iris_correlation.png` with a feature correlation heatmap

## Output

- **iris_exploration.png**: Multi-panel visualization including scatter plots, histograms, and box plots
- **iris_correlation.png**: Heatmap showing correlations between numerical features

## Dataset

The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Three species are included: setosa, versicolor, and virginica.
