# Unsupervised-Learning
## Clustering Process 
     Dimentionality Reduction - Feature Selection & Feature Extraction (PCA, ICA, Random Projection)
     Clustering Algorithm Selection & Tuning (K-Menas, Hierarchical, Density-Based, Gaussian-Mixture)
     Clustering Validation (External Indices, Internal Indices)
     Results interpratation
     
## Clustering Algorithm
### K-Means Clustering
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
      n_clusters: The number of clusters to form as well as the number of centroids to generate.
      init: ‘k-means++’: selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
            ‘random': choose k observations (rows) at random from data for the initial centroids.
            If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
      n_init: Thw number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best
      output of n_init consecutive runs in terms of inertia.
      
    Elbow method - to determine the optimal k 
    How K-Means works - 3 steps with the last two steps iterating until no change
    Feature Scaling - standardizing & minmax scaler
	
### Hierarchical Clustering
     single-link (the two closet points between clusters
     complete-link (the two farthest points between clusters 
     average-link (the average distance for between each pair points of two clusters 
     ward-method (the sum of the square distance of each point to the centroid of two clusters minus the sum of the square distance of
     each point to the self-centroid of each cluster)
     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
     Examples - clustering secreted protein families and to create clusters for different types of fungus
                diagram of human gastrotestinal microbiome 
     Advantages - hierarchical presentation is informative and provides an ability to visulize the relationship
                  especially potent when there is a real hierarchical relationship in the dataset (e.g. evolutionary biology)
     Disadvantages - snesitive to noise and outliers 
                     computationally intensive 

### Density Based Clustering
     DBSACN - based on each point to look up neighbor points within a specific distance (epsilon) and determined by the number of points
     in a neighborhood (min_samples) for a point to be considered as a core point and the total as a cluster including the point itself.
     It will label each point as noisy point (-1), border point (0), and core point (1).
     Examples - network traffic classification
                temperature anomaly detection 
     Advantages - no need to specify the number of clusters
                  flexible in the size and shape of clusters
		  able to exclude noise and outliers
     Disadvantages - border points may be included in either reachable cluster
                     facing difficulties finding clusters of varying densities
     
### Gaussian Mixture Models 
     https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
     Examples - nonparametric discovery of human routines from sensor data (e-reader, transportation velocity)
                computer vision to detect background and foreground
     Advantages - soft clustering (sample membership of multiple clustering)
                  flexible in cluster shape
     Disadvantages - sensitive to initialization values
                     possible to converge to a local optimum, not overall
		     slow convergence rate

## Cluster Validation
### External Indices 
     Adjusted Rand Index (-1, 1), the closer to 1, the better the clustering algorithm match to the labeled dataset
     = (any pair in the same cluster in both algorithm and labled dataset + 
     any pair in the different clusters in both algorithm and labeled dataset) / (N*(N-1)/2)
     https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
     http://faculty.washington.edu/kayee/pca/supp.pdf] 
         
### Internal Indices
     Silhouette coefficient, not a good indicator for density based clustering (DBSCAN), which uses DBCV as its internal index
     = the average of 
     (average distance to samples in the closest clusters b - average distance to other samples in the same cluster a) / max(a,b)]    

## Dimentionality Reduction
### Feature Selection
     Filter Methods - use a ranking or sorting algorithm to filter out those features that have less usefulness. Filter methods are
     based on discerning some inherent correlations among the feature data in unsupervised learning, or on correlations with the output
     variable in supervised settings. Filter methods are usually applied as a preprocessing step. Common tools for determining
     correlations in filter methods include: Pearson's Correlation, Linear Discriminant Analysis (LDA), and Analysis of Variance
     (ANOVA).
     
     Wrapper Methods - testing features impact on the performance of a model. The idea is to "wrap" this procedure around your
     algorithm, repeatedly calling the algorithm using different subsets of features, and measuring the performance of each model.
     Cross-validation is used across these multiple tests. The features that produce the best models are selected. Clearly this is a
     computationally expensive approach for finding the best performing subset of features, since they have to make a number of calls to
     the learning algorithm. Common examples of wrapper methods are: Forward Search, Backward Search, and Recursive Feature Elimination.
      https://scikit-learn.org/stable/modules/feature_selection.html

### Feature Extraction
#### Principal Component Analysis (PCA) - 
     Advantage: incorporate data from multiple features, and thus retain more information present in the various original inputs, than
     just losing that information by dropping many original inputs.
     Principal components are linear combinations of the original features in a dataset that aim to retain the most information in the
     original data.
     
     
#### Independent Component Analysis (ICA)
      
#### Random Projection
      
	
