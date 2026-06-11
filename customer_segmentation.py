import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("customer_data.csv")

def segment_customer(row):
    if row["AnnualIncome"] >= 50 and row["SpendingScore"] >= 50:
        return "High Value Customer"

    elif row["AnnualIncome"] >= 50 and row["SpendingScore"] < 50:
        return "Potential Customer"

    elif row["AnnualIncome"] < 50 and row["SpendingScore"] >= 50:
        return "Budget Spender"

    else:
        return "Low Value Customer"

df["Segment"] = df.apply(segment_customer, axis=1)

print("\nCustomer Segmentation Result:\n")
print(df)

segment_counts = df["Segment"].value_counts()

plt.figure(figsize=(8,5))
bars = plt.bar(segment_counts.index, segment_counts.values, color=["green", "orange", "blue", "red"])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')

plt.title("Customer Segmentation - Bar Graph")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.show()