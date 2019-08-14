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
     DBSACN - based on each point to look up neighbor points within a specific distance (epsilon) and determined by the number of points in a neighborhood (min_samples) for a point to be considered as a core point and the total as a cluster including the point itself. It will label each point as noisy point (-1), border point (0), and core point (1).
     Examples - network traffic classification
                temperature anomaly detection 
     Advantages - no need to specify the number of clusters
                  flexible in the size and shape of clusters
		  able to exclude noise and outliers
     Disadvantages - border points may be included in either reachable cluster
                     facing difficulties finding clusters of varying densities
     
## Gaussian Mixture Models and Cluster Validation
     https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
     Examples - nonparametric discovery of human routines from sensor data (e-reader, transportation velocity)
                computer vision to detect background and foreground
     Advantages - soft clustering (sample membership of multiple clustering)
                  flexible in cluster shape
     Disadvantages - sensitive to initialization values
                     possible to converge to a local optimum, not overall
		     slow convergence rate

## Clustering Validation
     External Indices - 
     Adjusted Rand Index (-1, 1), the closer to 1, the better the clustering algorithm match to the labeled dataset
     = (any pair in the same cluster in both algorithm and labled dataset + any pair in the different clusters in both algorithm and labeled dataset) / (N*(N-1)/2)
     http://faculty.washington.edu/kayee/pca/supp.pdf] 
         
     Internal Indices - 
     Silhouette coefficient 
     = the average of (average distance to samples in the closest clusters b - average distance to other samples in the same cluster a) / max(a,b)]
	
      relative indices
         compactness vs. separability
  ## Clustering Process 
     feature selection / extraction (PCA below)
     clustering algorithm selection & tuning   
     results interpratation
      		    
  

## Principal Component Analysis
   one of the most popular decomposition methods available today
   learn how matrix decomposition methods work conceptually
   apply principal component analysis to images of handwritten digits to reduce the dimensionality of these images
	
## Random Projection and Independent Component Analysis
   independent component analysis
   how this method can pull apart audio related to a piano, cello, and television that has been overlaid in the same file
