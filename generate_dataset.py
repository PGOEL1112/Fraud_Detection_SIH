import pandas as pd
import random

# List of common herbs
herbs = ["Amla", "Ashwagandha", "Tulsi", "Neem", "Giloy", "Triphala", "Shatavari", "Brahmi"]

# Initialize data dictionary
data = {
    "herb_type": [],
    "quality_score": [],
    "moisture_level": [],
    "stock_before": [],
    "stock_after": [],
    "amount": [],
    "Fraud": []
}

# Generate 1000 rows
for _ in range(1000):
    herb = random.choice(herbs)
    q_score = round(random.uniform(5.0, 12.0), 1)       # 5.0 to 12.0 quality score
    m_level = round(random.uniform(3.0, 9.0), 1)        # moisture 3% to 9%
    sb = random.randint(100, 500)                       # stock before
    sa = sb - random.randint(0, 10)                     # stock after (slightly less)
    amt = random.randint(2000, 20000)                   # transaction amount

    # Fraud rules for simulation
    # 1 = Fraud, 0 = Safe
    # Conditions: high quality (>10), large stock difference, very high amount
    fraud_flag = 1 if q_score > 10 or (sb - sa) > 5 or amt > 15000 else 0

    # Append to dictionary
    data["herb_type"].append(herb)
    data["quality_score"].append(q_score)
    data["moisture_level"].append(m_level)
    data["stock_before"].append(sb)
    data["stock_after"].append(sa)
    data["amount"].append(amt)
    data["Fraud"].append(fraud_flag)

# Create DataFrame
df = pd.DataFrame(data)

# Optional: save to CSV for retraining
# df.to_csv("herbal_transactions_1000.csv", index=False)

print("Large dataset with 1000 rows created successfully!")
print(df.head())

# ✅ Save the CSV filep
df.to_csv("herbal_transactions_1000.csv", index=False)

print("CSV file 'herbal_transactions_1000.csv' created successfully!")