# KDSmel.github.io

# Exploratory Geoscience Data Analysis: Multi-class Classification Problem


Although there are tons of great books and papers outside to practice machine learning, I always wanted to see something short, simple, and with a descriptive manuscript. I always wanted to see an example with an appropriate explanation of the procedure accompanied by detailed results interpretation. Model evaluation metrics should also need to be elaborated clearly.

In this work, I will try to include all important steps of ML modeling (even though some are not necessary for this dataset) to make a consistent and tangible example, especially for geoscientists. Eight important ML algorithms will be examined and results will be compared. I will try to have an argumentative model evaluation discussion. I will not go deep into the algorithm’s fundamentals.

To access the dataset and jupyter notebook find out my Git.
Note1: codes embedded in this manuscript are presented to understand the work procedure. If you want to exercise by yourself, I highly recommend using the [jupyter notebook file]([https://write.geeksforgeeks.org/](https://github.com/mardani72/Practical_ML_Tutorial_Facies_examp/blob/main/Part1_practical_Tut_ML_facies.ipynb)).
Note2: shuffling data can cause differences between your runs and what appears here.

This tutorial has four parts:
Part.1: Exploratory Data Analysis,
Part.2: Build Model & Validate,
Part.3: Model Evaluation-1,
Part.4: Model Evaluation-2

**1. Exploratory Data Analysis**
   1.1. Data visualization
      1–1–1 log-plot
      1–1–2 Bar plot
      1–1–3 Cross-plot
      
   1–2 Feature Engineering
      1–2–1 NaN imputation
      1–2–2 Feature extraction
      1–2–3 Oversampling
   1–3 Feature Importance
      1–3–1 Feature linear correlation
      1–3–2 Decision tree
      1–3–3 Permutation feature importance

**2- Build Model & Validate**
2–1 Baseline Model
2–2 Hyper-parameters
2–2–1 Grid search

**3- Model Evaluation-1**
3–1 Model metrics plot
3–2 Confusion matrix

**4- Model Evaluation-2**
4–1 Learning curves
4–2 ROC plot
4–3 Blind well prediction and evaluation

If you are totally fresh with python and ML concepts, you will need to get familiar with the basics to get advantages of this tutorial. As the dataset that we will work on here is a tabular CSV file including well logs and facies class, my two previous posts (10 steps in Pandas, 5 steps in Pandas) can be helpful for well log data handling, processing, and plotting. All implementation is based on scikit-learn libraries.

**Data Summary**
The dataset (facies_vectors.csv)for this study comes from Hugoton and Panoma Fields in North America which was used as class exercise at The University of Kansas (Dubois et. al, 2007). It consists of log data(the measurement of physical properties of rocks) of nine wells. We will use these log data to train supervised classifiers in order to predict discrete facies groups. For more detail, you may take a look here. The seven features are:

1. GR: this wireline logging tools measure gamma emission
2. ILD_log10: this is resistivity measurement
3. PE: photoelectric effect log
4. DeltaPHI: Phi is a porosity index in petrophysics.
5. PNHIND: Average of neutron and density log.
6. NM_M:nonmarine-marine indicator
7. RELPOS: relative position
   
The nine discrete facies (classes of rocks) are:

1. (SS) Nonmarine sandstone
2. (CSiS) Nonmarine coarse siltstone
3. (FSiS) Nonmarine fine siltstone
4. (SiSH) Marine siltstone and shale
5. (MS) Mudstone (limestone)
6. (WS) Wackestone (limestone)
7. (D) Dolomite
8. (PS) Packstone-grainstone (limestone)
9. (BS) Phylloid-algal bafflestone (limestone)
    
After data reading into python using Pandas, we can visualize it to understand data better. Before plotting, we need to define a color map(this step deserves to be in the Feature engineering part but we need here to plot color for facies classes) and devote color code for each facies.

**1–1 Data visualization*
1–1–1 log-plot

```python
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder
from collections import Counter
pd.set_option('display.max_rows', 30)
import numpy as np
import seaborn as sns

df = pd.read_csv('facies_vectors.csv')

# colors 
facies_colors = ['xkcd:goldenrod', 'xkcd:orange','xkcd:sienna','xkcd:violet',
       'xkcd:olive','xkcd:turquoise', "xkcd:yellowgreen", 'xkcd:indigo', 'xkcd:blue']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 
                 'MS',  'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
#establish facies label str    
df.loc[:,'FaciesLabels'] = df.apply(lambda row: label_facies(row, facies_labels), axis=1)
```


This is a function to create a plot.
```python
def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 6))
    ax[0].plot(logs.GR, logs.Depth, '-g',  alpha=0.8, lw = 0.9)
    ax[1].plot(logs.ILD_log10, logs.Depth, '-b',  alpha=0.8, lw = 0.9)
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-k',  alpha=0.8, lw = 0.9)
    ax[3].plot(logs.PHIND, logs.Depth, '-r',  alpha=0.8, lw = 0.9)
    ax[4].plot(logs.PE, logs.Depth, '-c',  alpha=0.8, lw = 0.9)
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((5*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
# call function to plot
make_facies_log_plot(
    data[data['Well Name'] == 'SHRIMPLIN'],
    facies_colors)
```

And the plot of the well SHRIMPLIN:
![image](./img1.png)

1–1–2 Bar plot

We can use the Counter function to evaluate each class contribution quantitatively. To see facies frequency distribution we can use a bar plot as:



This is an imbalanced dataset. Dolomite has the lowest member participation. Comparing coarse siltstone, dolomite appears 8 times less than that.

1–1–3 Cross plot

To visualize multiple pairwise bivariate distributions in a dataset, we may use the pairplot() function from the seaborn library. It shows the relationship for the combination of variables in the dataset in the matrix format with a univariate distribution plot in diagonal. It is clear that PE log has a non-linear relationship with average porosity. Other pairs do not show a clear pattern. The distribution pattern in diagonal shows that each label class (facies) with respect to each feature has acceptable separation although there is a strong overlap for various classes. The ideal pattern can be assumed as a clear separation of distribution plots in tall bell shape normal distribution graph.



Highlight: Collinear features are features that are highly correlated with each other. In machine learning, these lead to decreased generalization performance on the test set due to high variance and less model interpretability. In this dataset, we are not facing with collinearity. Using data.corr() command:


1–2 Feature Engineering
1–2–1 NaN imputation

It is common to have missing value in the dataset. To see the sum of null values for each column of features:

DataFrame.isna().sum()
# to find out which wells do not have PE
df_null = data_fe.loc[data_fe.PE.isna()]
df_null['Well Name'].unique()
#Categories (3, object): [ALEXANDER D, KIMZEY A, Recruit F9]

Here, PE has 917 null values.
There are several ways to deal with Null values in the dataset. The simplest approach is to drop the rows containing at least one null value. This can be logical with a bigger size dataset but in small data frames, single points are important. We can impute null values with mean or from adjacent data points in columns. Filling with mean value will not affect data variance and therefore will not have an impact on prediction accuracy, though can create data bias. Filling with the neighbor cells of column values can be appropriate if we have a geologically homogeneous medium like mass pure carbonate rocks.

Another approach, that I will implement here, to employe machine learning models to predict missing values. This is the best way of dealing with this dataset because we have just a single feature missing from the dataset, PE. On the other hand, filling with ML prediction is much better than the single mean value because we are able to see ML correlation and accuracy by dividing data to train and test sets.

Here, I will employ the Multi-Layer Perceptron Neural Network from scikit-learn to predict target value. I am not going to deep for this approach and use simply to predict missing values.



Predicted PE in well ALEXANDER D shows the normal range and variation. Prediction accuracy is 77%.

1–2–2 Feature Extraction

Having a limited set of features in this dataset can lead us to think about extracting some data from the existing dataset. First, we can convert the formation categorical data into numeric data. Our background knowledge can help us to guess that some facies are possibly present more in a specific formation rather than others. We can use the LabelEncoder function:

data_fe[‘Formation_num’] = LabelEncoder().fit_transform(data_fe[‘Formation’].astype(‘str’)) + 1
We converted formation category data into numeric to use as a predictor and added 1 to start predictor from 1 instead of zero. To see if new feature extraction would assist prediction improvement, we should define a baseline model then compare it with the extracted feature model.

Baseline Model Performance

For simplicity, we will use a logistic regression classifier as a baseline model and will examine model performance with a cross-validation concept. Data will be split into 10 subgroups and the process will be repeated 3 times.


Here, we can explore whether feature extraction can improve model performance. There are many approaches while we will use some transforms for chaining the distribution of the input variables such as Quantile Transformer and KBins Discretizer. Then, will remove linear dependencies between the input variables using PCA and TruncatedSVD. To study more refer here.
Using feature union class, we will define a list of transforms to perform results aggregated together. This will create a dataset with lots of feature columns while we need to reduce dimensionality to faster and better performance. Finally, Recursive Feature Elimination, or RFE, the technique can be used to select the most relevant features. We select 30 features.


Accuracy improvement shows that feature extraction can be a useful approach when we are dealing with limited features in the dataset.

1–2–3 Oversampling

In imbalanced datasets, we can use the resampling technique to add some more data points to increase members of minority groups. This can be helpful whenever minority label targets have special importance such as credit card fraud detection. In that example, fraud can happen with less than 0.1 percent of transactions while it is important to detect fraud.
In this work, we will add pseudo observation for the Dolomite class which has the lowest population

Synthetic Minority Oversampling Technique, SMOTE: the technique is used to select nearest neighbors in the feature space, separate examples by adding a line, and producing new examples along the line. The method is not merely generating the duplicates from the outnumbered class but applied K-nearest neighbors to generate synthetic data.


Accuracy improved by 3 percent but in multi-class classification, accuracy is not the best evaluation metric. We will cover others in the part.3.

1–3 Feature Importance
Some machine learning algorithms (not all) offer an importance score to help the user to select the most efficient features for prediction.

1–3–1 Feature linear correlation

The concept is simple: features have a higher correlation coefficient with target values are important for prediction. We can extract these coef’s like:



1–3–2 Decision tree

This algorithm provides importance scores based on the reduction in the criterion used to split in each node such as entropy or Gini.



1–3–3 Permutation feature importance

Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular. This is especially useful for non-linear or opaque estimators. The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled.



In all these feature importance plots we can see that predictor number 6 (PE log) has the most importance in label prediction. Based on the model that we select to evaluate the result, we may choose features based on their importance and neglect the rest to speed up the training process. This is very common if we are rich in feature quantity, though in our example dataset here, we will use all features as predictors are limited.

Summary
Data preparation is one of the most important and time-consuming steps in machine learning. Data visualization can help us to understand data nature, borders, and distribution. Feature engineering is required especially if we have null and categorical values. In small datasets, feature extraction and oversampling can be helpful for model performances. Finally, we can analyze features in the dataset to see the importance of features for different model algorithms.
