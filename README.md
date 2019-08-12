# Unsupervised-Learning
### Clustering - which groups data together based on similarities
### Dimensionality Reduction - which condenses a large number of features into a (usually much) smaller set of features

## Clustering
## K-Means Clustering
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
      n_clusters: The number of clusters to form as well as the number of centroids to generate.
      init: ‘k-means++’: selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
            ‘random': choose k observations (rows) at random from data for the initial centroids.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
      n_init: Thw number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
      
    Elbow method - to determine the optimal k 
    How K-Means works - 3 steps with the last two steps iterating until no change
    Feature Scaling - standardizing & minmax scaler
	
## Hierarchical Clustering
     single-link (the two closet points between clusters
     complete-link (the two farthest points between clusters 
     average-link (the average distance for between each pair points of two clusters 
     ward-method (the sum of the square distance of each point to the centroid of two clusters minus the sum of the square distance of each point to the self-centroid of each cluster)
     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
     Examples - clustering secreted protein families and to create clusters for different types of fungus
                diagram of human gastrotestinal microbiome 
     Advantages - hierarchical presentation is informative and provides an ability to visulize the relationship
                  especially potent when there is a real hierarchical relationship in the dataset (e.g. evolutionary biology)
     Disadvantages - snesitive to noise and outliers 
                     computationally intensive 

## Density Based Clustering
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
