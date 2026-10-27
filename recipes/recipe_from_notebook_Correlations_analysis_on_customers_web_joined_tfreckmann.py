
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Analyzing correlations betweeen variables in customers_web_joined
# 
# In this notebook, we are going to study the correlations.
# 
# * [Setup and loading the data](#setup)
# * [Correlation matrix](#corr-matrix)
# * [Scatterplot matrix](#scatter-matrix)
# * [Detailed analysis between two variables](#two-vars)
# 
# <center><strong>Select Cell > Run All to execute the whole analysis</strong></center>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Setup and dataset loading <a id="setup"></a>
# 
# First of all, let's load the libraries that we'll use

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# %pylab inline
import numpy as np
import matplotlib.pyplot as plt
import dataiku                          # Access to Dataiku datasets
import pandas as pd, numpy as np        # Data manipulation
import scipy.cluster.hierarchy as sch   # Used for reordering the correlation matrix
import warnings                         # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# The first thing we do is now to load the dataset and put aside the three main types of columns:
# 
# * Numerics
# * Categorical
# * Dates
# 
# Since analyzing correlations requires to have the data in memory, we are only going to load a sample of the data. Modify the following cell to change the size of the sample.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Take a handle on the dataset
mydataset = dataiku.Dataset("customers_web_joined")

# Load the first 100'000 lines.
# You can also load random samples, limit yourself to some columns, or only load
# data matching some filters.
#
# Please refer to the Dataiku Python API documentation for more information
df = mydataset.get_dataframe(
    limit = 100000)

# Get the column names
numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
categorical_columns = list(df.select_dtypes(include=[object]).columns)
date_columns = list(df.select_dtypes(include=['<M8[ns]']).columns)

# Print a quick summary of what we just loaded
print("Loaded dataset")
print("   Rows: %s" % df.shape[0])
print("   Columns: %s (%s num, %s cat, %s date)" % (df.shape[1],
                                                    len(numerical_columns), len(categorical_columns),
                                                    len(date_columns)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Correlation matrix <a id="corr-matrix"></a>
# 
# The very first correlation analysis consists of plotting the "Correlation matrix" for numerical variables.
# 
# For each couple of numerical variables, this computes the "strength" of the correlation (called the Pearson coefficient):
# 
#  * 1.0 means a perfect correlation
#  * 0.0 means no correlation
#  * -1.0 means a perfect "inverse" correlation
# 
# Since it does not really make sense to print this correlation plot for hundred of variables, we are restricting it to the first 50 numerical variables of the dataset. Modify the following cell to change this

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Select variables to plot for the correlation matrix
corr_matrix_vars = numerical_columns[0:50]

print("Plotting the correlation matrix on the following variables : %s" % corr_matrix_vars)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Only select the requested columns
df_corr_matrix = df[corr_matrix_vars]

# This computes the Pearson coefficient for all couples
corr = df_corr_matrix.corr().fillna(0)

# Start drawing

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
size = max(10, len(corr.columns)/2.)
f, ax = plt.subplots(figsize=(size, size))

# Draw the heatmap with the mask and correct aspect ratio
p = ax.imshow(corr.mask(mask), cmap='inferno')
f.colorbar(p)
ax.set_yticks(range(0, len(corr.columns)))
ax.set_yticklabels(corr.columns)
ax.set_xticks(range(0, len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation = 90)
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Reordered correlation matrix
# 
# An interesting improvement over the correlation matrix is to reorder it by similarity between the variables so that the "groups" of variables that are strongly correlated appear close in the matrix.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Generate features and distance matrix.
D = corr.values
# Compute and plot dendrogram.
Y = sch.linkage(D, method='centroid')
Z = sch.dendrogram(Y, orientation='right',no_plot=True)
# Compute distance matrix.
index = Z['leaves']
D = D[index,:]
D = D[:,index]

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
size = max(10, len(corr.columns)/2.)
f, ax = plt.subplots(figsize=(size, size))

# Draw the heatmap with the mask and correct aspect ratio

p = ax.imshow(np.where(mask, float('NaN'), D), cmap='inferno')
f.colorbar(p)
ax.set_yticks(range(0, len(corr.columns)))
ax.set_yticklabels(corr.columns[index])
ax.set_xticks(range(0, len(corr.columns)))
ax.set_xticklabels(corr.columns[index], rotation = 90)
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Scatter matrices <a id="scatter-matrix"></a>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Only generate the scatterplot matrix on a sample
df_scatter_samp = df.sample(min(5000, df.shape[0])) # 5000 points maximum on the scatter plot

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
scatter_matrix_vars = ["revenue", "price_first_item_purchased", "age", "pages_visited"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig, axs = plt.subplots(ncols=len(scatter_matrix_vars), nrows=len(scatter_matrix_vars), figsize=(16, 16))
for i in range(0, len(scatter_matrix_vars)):
    for j in range(0, len(scatter_matrix_vars)):
        ax = axs[j, i]
        x_s = df_scatter_samp[scatter_matrix_vars[i]]
        y_s = df_scatter_samp[scatter_matrix_vars[j]]
        if i == j:
            ax.hist(x_s)
            ax.set_yticks([])
        else:
            ax.scatter(x_s, y_s)
            if i > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(scatter_matrix_vars[j])
        if i == 0:
            ax.set_ylabel(scatter_matrix_vars[j])
        if j == len(scatter_matrix_vars) - 1:
            ax.set_xlabel(scatter_matrix_vars[i])
# align X axis
for i in range(0, len(scatter_matrix_vars)):
    for j in range(0, len(scatter_matrix_vars)):
        axs[j, i].set_xlim(axs[i, i].get_xlim())
plt.show()
from dataiku import insights
insights.save_figure("PairGrid", fig)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
ipg13cw0 = dataiku.Folder("IPG13Cw0")
ipg13cw0_info = ipg13cw0.get_info()