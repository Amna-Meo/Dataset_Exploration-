import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 60)
print("IRIS DATASET EXPLORATION")
print("=" * 60)

# 1. Basic Information
print("\n1. Dataset Shape:")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. Column Names:")
print(f"   {list(df.columns)}")

print("\n3. First 5 Rows:")
print(df.head())

print("\n4. Dataset Info:")
print(df.info())

print("\n5. Statistical Summary:")
print(df.describe())

# 2. Visualizations
fig = plt.figure(figsize=(16, 12))

# Scatter plots
print("\n" + "=" * 60)
print("Creating visualizations...")
print("=" * 60)

# Scatter plot 1: Sepal length vs Sepal width
plt.subplot(3, 3, 1)
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species, alpha=0.6)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot 2: Petal length vs Petal width
plt.subplot(3, 3, 2)
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], label=species, alpha=0.6)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()
plt.grid(True, alpha=0.3)

# Histograms
plt.subplot(3, 3, 4)
df['sepal length (cm)'].hist(bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram: Sepal Length')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
df['sepal width (cm)'].hist(bins=20, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram: Sepal Width')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
df['petal length (cm)'].hist(bins=20, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram: Petal Length')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 7)
df['petal width (cm)'].hist(bins=20, edgecolor='black', alpha=0.7, color='red')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.title('Histogram: Petal Width')
plt.grid(True, alpha=0.3)

# Box plots
plt.subplot(3, 3, 8)
df.boxplot(column='sepal length (cm)', by='species', ax=plt.gca())
plt.title('Box Plot: Sepal Length by Species')
plt.suptitle('')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')

plt.subplot(3, 3, 9)
df.boxplot(column='petal length (cm)', by='species', ax=plt.gca())
plt.title('Box Plot: Petal Length by Species')
plt.suptitle('')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

plt.tight_layout()
plt.savefig('iris_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to 'iris_exploration.png'")

# Additional: Correlation heatmap
fig2 = plt.figure(figsize=(8, 6))
correlation = df.drop('species', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved to 'iris_correlation.png'")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
