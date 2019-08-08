# Unsupervised-Learning
### Clustering - which groups data together based on similarities
### Dimensionality Reduction - which condenses a large number of features into a (usually much) smaller set of features

## Clustering
####   K-Means
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
n_clusters : int, optional, default: 8 - The number of clusters to form as well as the number of centroids to generate.
init : {‘k-means++’, ‘random’ or an ndarray} - Method for initialization, defaults to ‘k-means++’:
‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
‘random’: choose k observations (rows) at random from data for the initial centroids.
If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
n_init : int, default: 10 - Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
	
## Hierarchical and Density Based Clustering
   density based 'closeness' measures
   how this can be used in traffic classification
   anomaly detection - finding points that aren't like others in your dataset
	 
## Gaussian Mixture Models and Cluster Validation
   gaussian mixture models
   last of the clustering algorithms you will learn before moving to matrix decomposition methods

## Principal Component Analysis
   one of the most popular decomposition methods available today
   learn how matrix decomposition methods work conceptually
   apply principal component analysis to images of handwritten digits to reduce the dimensionality of these images
	
## Random Projection and Independent Component Analysis
   independent component analysis
   how this method can pull apart audio related to a piano, cello, and television that has been overlaid in the same file
