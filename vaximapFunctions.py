'------------Import Modules------------'

# DS Modules
import random
import datetime
import pandas as pd
import numpy as np
import itertools

# Visualisation
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import seaborn as sns

# Statistical Tests
from scipy.stats import normaltest, ttest_ind, chi2_contingency, chi2

# Clustering
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import dendrogram


'------------ Functions ------------'

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