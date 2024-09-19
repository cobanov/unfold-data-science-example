[Home](../bootcamp-main.md)

> “Divide each difficulty into as many parts as is feasible and necessary to resolve it.”  
> — René Descartes

# Day 22: Clustering Algorithms - K-Means, Hierarchical Clustering

Clustering is an unsupervised learning technique used to group similar data points into clusters. It is widely used in data exploration, customer segmentation, image analysis, and more. Today, we will explore two popular clustering algorithms: K-Means and Hierarchical Clustering. We will learn their concepts, implementation using Python's scikit-learn library, and how to evaluate clustering results.

- Introduction to clustering
- K-Means clustering: theory, implementation, and evaluation
- Hierarchical clustering: theory, implementation, and evaluation
- Visualizing clustering results

## Clustering

Clustering is a fundamental task in unsupervised learning, where the goal is to divide a dataset into groups (or _clusters_) such that points within the same cluster are more similar to each other than to points in other clusters. This technique is used in various fields such as customer segmentation, anomaly detection, image processing, and more. Unlike supervised learning, clustering algorithms don’t rely on labeled data. Instead, they work by discovering patterns and relationships within the data itself.

Formally, clustering is defined as finding a partition of the dataset $` X = \{x_1, x_2, \dots, x_n\} `$ into $` K `$ clusters $` C = \{C_1, C_2, \dots, C_K\} `$, such that points within the same cluster are similar according to a specific similarity or distance metric (commonly Euclidean distance). Different clustering algorithms define "similarity" in different ways, and the choice of algorithm depends on the structure and distribution of the data.

## K-Means Clustering

**K-Means Clustering** is one of the most widely used clustering algorithms due to its simplicity and efficiency. The core idea of K-Means is to partition the data into $` K `$ clusters, where each data point belongs to the cluster with the nearest mean. It’s a centroid-based clustering method, and the process involves iterative refinement of the cluster centroids to minimize the within-cluster variance.

Given a dataset $` X \in \mathbb{R}^n `$, the algorithm starts by selecting $` K `$ initial centroids (randomly or through strategies like K-Means++). The algorithm then alternates between two main steps:

1. **Assignment Step**: Assign each data point to the nearest centroid. For each point $` x_i `$, the nearest centroid $` \mu_j `$ is found using the Euclidean distance:

```math
\text{arg min}\_j \| x_i - \mu_j \|^2
```

This step partitions the data into clusters $` C_j `$, where each cluster contains points closer to its centroid than any other centroid.

2. **Update Step**: Once all points have been assigned, the centroids are updated by computing the mean of the points in each cluster:

```math
\mu*j = \frac{1}{|C_j|} \sum*{x_i \in C_j} x_i
```

These new centroids are then used in the next iteration of the algorithm.

The algorithm continues alternating between these two steps until the centroids no longer change (or the change is below a predefined threshold), or a maximum number of iterations is reached. The objective of K-Means is to minimize the total within-cluster variance, given by the following cost function:

```math
J = \sum*{j=1}^{K} \sum*{x_i \in C_j} \| x_i - \mu_j \|^2
```

K-Means is guaranteed to converge to a local minimum, but its performance depends on the initial placement of centroids, which is why multiple runs with different initializations (or the K-Means++ initialization) are recommended.

### Implementation of K-Means

In Python, K-Means is implemented using the `KMeans` class from the `scikit-learn` library. A basic implementation looks like this:

```python
from sklearn.cluster import KMeans

#Assuming X is the dataset
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_  # Cluster labels for each point
centroids = kmeans.cluster_centers_  # Centroid locations
```

### Evaluation of K-Means

There are several ways to evaluate the quality of K-Means clustering:

1. **Within-Cluster Sum of Squares (WCSS)**: This is the cost function $` J `$ mentioned earlier. Lower WCSS values indicate better clustering.
2. **Silhouette Score**: This measures how similar points are to their own cluster compared to other clusters. It ranges from -1 to 1, with values closer to 1 indicating better-defined clusters.

```math
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
```

where $` a(i) `$ is the average distance between $` x_i `$ and other points in the same cluster, and $` b(i) `$ is the minimum average distance between $` x_i `$ and points in a different cluster. 3. **Elbow Method**: This method helps determine the optimal number of clusters $` K `$. It involves plotting WCSS against different values of $` K `$, and looking for the "elbow point" where adding more clusters doesn't significantly reduce WCSS.

## Hierarchical Clustering

**Hierarchical Clustering** is another popular clustering method, but unlike K-Means, it doesn’t require the user to specify the number of clusters in advance. Instead, it builds a hierarchy of clusters, which can be visualized using a **dendrogram**. Hierarchical clustering comes in two main forms: **agglomerative** (bottom-up) and **divisive** (top-down).

### Agglomerative Hierarchical Clustering

Agglomerative clustering starts with each data point as its own cluster and then successively merges the closest clusters until all points belong to one single cluster. The distance between clusters can be defined in various ways:

1. **Single Linkage**: The distance between two clusters is defined as the minimum distance between any two points in the clusters:

```math
D(C*i, C_j) = \min*{x \in C_i, y \in C_j} \|x - y\|
```

2. **Complete Linkage**: The distance is defined as the maximum distance between any two points in the clusters:

```math
D(C*i, C_j) = \max*{x \in C_i, y \in C_j} \|x - y\|
```

3. **Average Linkage**: The distance is the average distance between all points in the two clusters:

```math
D(C*i, C_j) = \frac{1}{|C_i| |C_j|} \sum*{x \in C*i} \sum*{y \in C_j} \|x - y\|
```

4. **Ward's Method**: This method minimizes the variance between clusters. It merges the two clusters that result in the smallest increase in the total within-cluster variance.

The result of agglomerative clustering is a tree-like structure (dendrogram), where each merge can be represented as a horizontal line. The height of the line corresponds to the distance between the merged clusters, and by cutting the dendrogram at a particular level, you can choose the number of clusters.

### Implementation of Hierarchical Clustering

In Python, hierarchical clustering can be implemented using the `AgglomerativeClustering` class from `scikit-learn` and the `dendrogram` function from `scipy`:

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Assuming X is the dataset
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hierarchical.fit_predict(X)

# For creating the dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
plt.show()
```

### Evaluation of Hierarchical Clustering

Evaluation of hierarchical clustering can be done using the **cophenetic correlation coefficient**, which measures how faithfully the dendrogram represents the pairwise distances between the original data points. Silhouette scores and dendrogram analysis can also be used to evaluate the quality of the clusters.

The **dendrogram** itself can provide insights into the clustering structure. By visualizing it, you can decide the number of clusters by cutting the tree at different levels, making hierarchical clustering a flexible method when the number of clusters is not known a priori.

K-Means and Hierarchical Clustering are both powerful clustering methods with distinct strengths. K-Means is computationally efficient and works well with large datasets when the number of clusters is known, while Hierarchical Clustering is more flexible and provides deeper insights into the structure of the data but at a higher computational cost. Each method has its own evaluation metrics, making it essential to choose the right one based on the problem at hand.

## Problem

**Problem Statement:** You are given a dataset containing information about customers, including their annual income and spending score. Your tasks are:

1. Implement K-Means clustering to segment customers.
2. Implement Hierarchical clustering to segment customers.
3. Visualize the clustering results.

**Dataset:**

```csv
CustomerID, Annual Income (k$), Spending Score (1-100)
1, 15, 39
2, 16, 81
3, 17, 6
4, 18, 77
5, 19, 40
6, 20, 76
7, 21, 6
8, 22, 94
9, 23, 3
10, 24, 72
```

## Explanation

1. **Create DataFrame:** We create a DataFrame from the given dataset containing customer information.
2. **Select Features:** We select the features (Annual Income and Spending Score) for clustering.
3. **Implement K-Means Clustering:** We use the K-Means algorithm with `k=3` to segment customers into clusters. We add the cluster labels to the DataFrame and visualize the clustering results using a scatter plot.
4. **Implement Hierarchical Clustering:** We use the Agglomerative Clustering algorithm with `n_clusters=3` to segment customers into clusters. We add the cluster labels to the DataFrame and visualize the clustering results using a scatter plot.
5. **Create Dendrogram:** We create a dendrogram using the linkage matrix to visualize the hierarchical clustering process and understand the merging of clusters.

## Solution

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

# Step 1: Create a DataFrame from the dataset
data = pd.read_csv("./dataset/customer_dataset.csv")
df = pd.DataFrame(data)

# Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 2: Implement K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

# Visualize K-Means clustering results
plt.figure(figsize=(10, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label="Cluster")
plt.show()

# Step 3: Implement Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X)

# Visualize Hierarchical clustering results
plt.figure(figsize=(10, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Hierarchical_Cluster"],
    cmap="viridis",
)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label="Cluster")
plt.show()

# Step 4: Create dendrogram for Hierarchical clustering
Z = linkage(X, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer ID")
plt.ylabel("Distance")
plt.show()

```

By working through this exercise, you will learn how to implement and visualize K-Means and Hierarchical Clustering algorithms, helping you segment data points into meaningful clusters and gain insights from your data.

---

[K-Nearest Neighbors ← ](../bootcamp-subpages/day-21.md) | [Home](../bootcamp-main.md) | [→ Dimensionality Reduction](../bootcamp-subpages/day-23.md)

---
