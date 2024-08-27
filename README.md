# Search Ads Optimization

This project focuses on optimizing search advertisements using machine learning techniques. The goal is to improve ad targeting and maximize Return on Investment (ROI) by predicting conversion rates based on various features.

## Project Overview

The project demonstrates the process of building a predictive model for search ad conversion rates using a dataset that includes features such as ad category, budget, clicks, and impressions. The model is trained using a Random Forest Regressor, and its performance is evaluated using metrics like Mean Squared Error (MSE) and R² Score.

## Features

- **Data Preprocessing**: The data is cleaned, missing values are handled, categorical variables are encoded, and numerical features are scaled to prepare the data for modeling.
- **Model Training**: A Random Forest Regressor model is used to predict conversion rates.
- **Model Evaluation**: The model is evaluated using MSE and R² Score to measure its accuracy.
- **Visualization**: A scatter plot is generated to compare actual vs. predicted conversion rates.

## Project Structure

- `ads_optimization.py`: Main script for loading data, preprocessing, training the model, and visualizing results.
- `create_dataset.py`: Script to generate a synthetic dataset (if applicable).
- `search_ads_data.csv`: Example dataset used in the project.
- `README.md`: Project documentation.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `venv/`: Virtual environment directory (not included in the repository).

## Installation and Setup

### Prerequisites

- Python 3.x
- Git

### Clone the Repository

```bash
git clone https://github.com/Viraj-P/Search-Ads-Optimization.git
cd Search-Ads-Optimization
