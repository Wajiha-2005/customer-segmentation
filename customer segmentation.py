import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv("customer dataset.csv")
X = df[['AnnualIncome', 'SpendingScore']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

summary = df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean()
print("\nCluster Summary:")
print(summary)
segment_names = {
    0: "Budget Customers",
    1: "Premium Customers",
    2: "Regular Customers"
}

df['CustomerSegment'] = df['Cluster'].map(segment_names)


print("\nCustomer Segments:")
print(df[['CustomerID', 'AnnualIncome', 'SpendingScore', 'CustomerSegment']])

segment_counts = df['CustomerSegment'].value_counts()
plt.figure(figsize=(8,5))

bars = plt.bar(
    segment_counts.index,
    segment_counts.values,
    color=['green', 'orange', 'blue']
)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        str(int(height)),
        ha='center',
        va='bottom',
        fontsize=11
    )

plt.title("Customer Segmentation Analysis")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()