# Libraries
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Import the mtcars dataset from the web + keep only numeric variables
url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
df = pd.read_csv(url)
df = df.set_index('model')
del df.index.name
df

# Calculate the distance between each sample
# You have to think about the metric you use (how to measure similarity) + about the method of clusterization you use (How to group cars)
Z = linkage(df, 'average')

# Make the dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (Average)')
dendrogram(Z, labels=df.index, leaf_rotation=90)


# Now compute KMeans and print the vehicles in each cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(df)

print('Class 1:')
for i in range(len(y_kmeans)):
    if y_kmeans[i] == 0:
        print(df.index[i])
print('Class 2:')
for i in range(len(y_kmeans)):
    if y_kmeans[i] == 1:
        print(df.index[i])
print('Class 3:')
for i in range(len(y_kmeans)):
    if y_kmeans[i] == 2:
        print(df.index[i])