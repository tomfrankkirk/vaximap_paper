# helper functions to perform analysis of the vaximap dataset
'------------Import Modules------------'

# DS Modules
import random
import datetime
import pandas as pd
import numpy as np
import itertools
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Tests
from scipy.stats import normaltest, ttest_ind, chi2_contingency, chi2

# Clustering
from sklearn import cluster
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import dendrogram


'------------ Functions ------------'

def iterative_nn_clusters(lat_long, max_cluster_size, merge=False):
    """
    Iterative nearest-neighbour clustering algorithm with constraint
    on maximum cluster size. 

    Args: 
        lat_long (np.array): N x 2 array of coordinates 
        max_cluster_size (int): limit on cluster size 

    Returns: 
        np.array size N of integer labels in the range (0,C), where
            C is the number of clusters generated
    """
    
    # number of clusters implied by cluster size 
    n_clusters = np.ceil(lat_long.shape[0] / max_cluster_size).astype(int)

    # k-means to derive cluster centroids (note that we don't actually use the 
    # clustering itself, just the centroids)
    centroids = ((cluster.KMeans(n_clusters).fit(lat_long)).cluster_centers_)

    # For each point, calculate distance to all centroids.
    # Then calculate for each the distance (smallest dist - largest dist), which is < 0
    # Sort the array ascending, from most negative to least. Points that are only close 
    # to one cluster will get a very negative cost, so they are sorted first 
    dists = np.linalg.norm(np.array([lat_long - c for c in centroids]), ord=2, axis=-1).T
    min_max_dist = np.array([ d.min() - d.max() for d in dists ])
    sorted_inds = np.argsort(min_max_dist)

    # We now work through the points in order and assign each to its closest centroid
    # that is not already full 
    clusters = -1 * np.ones(sorted_inds.size, dtype=int)
    for ind in sorted_inds: 
        cents_ord = np.argsort(dists[ind,:])
        for cent in cents_ord: 
            if (clusters == cent).sum() < max_cluster_size:
                clusters[ind] = cent
                break

    # Merge small clusters into larger ones (NB this doesn't remove small
    # ones entirely, just maximises number of full clusters)
    if merge: 

        # if there are at least 2 small clusters existing
        while (np.bincount(clusters) < max_cluster_size).sum() > 1: 

            # we work in indices of all clusters, ie, 0 to N-1
            # find the smallest under-sized cluster
            counts = np.bincount(clusters)
            to_merge = np.flatnonzero(counts < max_cluster_size)
            merge_from = to_merge[np.argmin(counts[to_merge])]

            # find the other undersized cluster that is closest to this one, 
            # we will merge into this one 
            dists_to_others = np.linalg.norm(centroids[merge_from,:] - centroids[to_merge], ord=2, axis=-1)
            merge_to = to_merge[np.argsort(dists_to_others)[1]]

            # whilst the destination cluster isn't full, we find all points that need
            # reassigning, find the one that is closest to the destination cluster, 
            # and reassign it. break immediately and start the whole exercise again 
            while (clusters == merge_to).sum() < max_cluster_size: 
                to_reassign = np.flatnonzero(clusters == merge_from)
                reassign_dists = np.linalg.norm(lat_long[to_reassign,:] - centroids[merge_to], ord=2, axis=-1)
                closest = to_reassign[np.argmin(reassign_dists)]
                clusters[closest] = merge_to
                break 

    # Sanity checks: no cluster is too large, all points got a label, 
    # and the labels are 0-indexed (so the max label is N-1)
    assert np.bincount(clusters).max() <= max_cluster_size
    assert (clusters > -1).all()
    assert (clusters < n_clusters).all()

    return clusters


def generate_vaximap_solution(points, cluster_size):
    clusters = iterative_nn_clusters(points, cluster_size, merge=True)
    routes = []
    distance = 0 
    for cidx in np.unique(clusters):
        cluster_inds = np.flatnonzero(clusters==cidx)
        ps = points[cluster_inds,:]
        problem = mlrose.TravellingSales(coords=ps)
        optimisation = mlrose.TSPOpt(length=ps.shape[0], fitness_fn=problem)
        route, cost = mlrose.genetic_alg(optimisation, mutation_prob=0.25, 
                                         max_attempts=50)
        routes.append(cluster_inds[route])
        distance += cost 
    
    return routes, distance 


def FindMaxLength(lst):
    """
    Returns max length of list elements in nested list.
    Params:
        lst (type: list): nested list
    """

    maxList = max(lst, key = len)
    maxLength = max(map(len, lst))
      
    return maxLength


def categoricalCountPlot(dataframe, categorical_feature):
    # generate count plots
    plt.figure(figsize=(15,4))
    ax = sns.countplot(y=categorical_feature, data=dataframe)
    ax.set_title("Count plot of {}".format(categorical_feature))
    return plt.show()


def univariateCategorical(dataframe, categorical_features):
    """
    Functions returns plots of feature categories
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        categorical_features: type(List), list of column names for categorical features
    """
    for categorical_feature in categorical_features:
        
        # extract majority class over entire proportion
        print("{f}: \nNumber of distinct categories = {d} \nMajority class = '{a}' with {b} ({p}%) of samples."\
              .format(f=categorical_feature,
                      a=dataframe[categorical_feature].mode()[0], d=dataframe[categorical_feature].nunique(),
                      b=len(dataframe.loc[dataframe[categorical_feature] == dataframe[categorical_feature].mode()[0]]),
                      p=round(100*len(dataframe.loc[dataframe[categorical_feature] == dataframe[categorical_feature].mode()[0]])/len(dataframe))))
        
        # extract number of nulls
        print("Number of nulls in feature = {a} ({b}%)"\
              .format(a=len(dataframe[categorical_feature]\
                            .loc[dataframe[categorical_feature].isnull()]),
                     b=round(100 * len(dataframe[categorical_feature]\
                                       .loc[dataframe[categorical_feature].isnull()])/len(dataframe), 1)))
        
        # plot the count plot
        categoricalCountPlot(dataframe, categorical_feature)
    
    return



def univariateContinuous(dataframe, numerical_features, n_bins=20):
    """
    Functions returns summary dataframe of numerical features and histogram plots
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        numerical_features: type(List), list of column names for continuous features
    """

    # generate plots of histograms
    fig, axs = plt.subplots(1, len(numerical_features), sharey=False, sharex=False, tight_layout=True, figsize=(20,8))
    axs = axs.ravel()
    # We can set the number of bins with the `bins` kwarg
    for idx,ax in enumerate(axs):
        ax.grid()
        ax.set_title("Histogram of {}".format(numerical_features[idx]))
        ax.hist(dataframe[numerical_features[idx]], bins=n_bins, color=["red"])

    # create feature summary and include some additional metrics for median, skew, kurtosis and normality test
    summary_df = dataframe[numerical_features].describe()\
    .round()\
    .append(pd.DataFrame(columns= ["index"] + numerical_features,\
                        data=[["median"] + [round(dataframe[cols].median()) for cols in numerical_features],
                            ["skew"] + [round(dataframe[cols].skew(), 2) for cols in numerical_features],
                            ["kurtosis"] + [round(dataframe[cols].kurtosis(), 2) for cols in numerical_features],
                             ["normality: statistic"] + [round(normaltest(dataframe[cols]).statistic, 2) \
                                                         for cols in numerical_features],
                             ["normality: p-value"] + [round(normaltest(dataframe[cols]).pvalue, 2) \
                                                       for cols in numerical_features]])\
    .set_index("index"))
    return summary_df


def bivariateContinuous(dataframe, numerical_feature, target, corr_type="spearman"):
    """
    Functions returns boxplot of numerical_feature vs target feature
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        numerical_feature: column names for continuous feature
        target: y feature comparing to
    """
    # get correlation between two features
    correlation = dataframe[target].corr(dataframe[numerical_feature], corr_type)
    # generate figure 
    plt.figure(figsize=(15,7))
    # generate boxplots
    plt.title("{y} vs. {x}, Correlation = {c}".format(x=numerical_feature, y=target, c=round(correlation, 2)))
    sns.boxplot(y=dataframe[numerical_feature], x=dataframe[target])

    return


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)