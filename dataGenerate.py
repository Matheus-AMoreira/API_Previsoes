import pandas as pd
import numpy as np
import random

def GenerateData(csv_path):
    # Create monthly date range from Jan 2010 to Dec 2024
    dates = pd.date_range(start="2010-01", end="2024-12", freq="MS")
    years = dates.year
    months = dates.month
    n_months = len(dates)

    # Generate units sold with a downward trend (400 to 320) and some randomness
    trend = np.linspace(400, 320, n_months)
    units = [int(trend[0])]
    for i in range(1, n_months):
        base = trend[i]
        prev = units[-1]
        lower = max(int(prev * 0.5), int(base * 0.9))
        upper = min(int(prev * 2.0), int(base * 1.1))
        units.append(random.randint(lower, upper))

    # Generate price per unit from R$15 to R$30 over time
    prices = np.round(np.linspace(15.00, 30.00, n_months), 2)

    # Create DataFrame
    df = pd.DataFrame({
        "Year": years,
        "Month": months,
        "Units_Sold": units,
        "Price_BRL": prices
    })

    # Save to CSV
    df.to_csv(csv_path, index=False)

GenerateData('Dados/toy_sales_2010-2024.csv')