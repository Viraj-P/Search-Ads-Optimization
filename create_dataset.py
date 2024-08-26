import pandas as pd
import numpy as np

# Create a synthetic dataset
np.random.seed(42)

data = {
    'ad_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], size=1000),
    'budget': np.random.uniform(100, 1000, size=1000),
    'clicks': np.random.poisson(lam=50, size=1000),
    'impressions': np.random.poisson(lam=200, size=1000),
    'conversion_rate': np.random.uniform(0.01, 0.1, size=1000)
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('search_ads_data.csv', index=False)
