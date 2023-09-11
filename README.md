# Exploratory Geoscience Data Analysis: Multi-class Classification Problem


Although there are tons of great books and papers outside to practice machine learning, I always wanted to see something short, simple, and with a descriptive manuscript. I always wanted to see an example with an appropriate explanation of the procedure accompanied by detailed results interpretation. Model evaluation metrics should also need to be elaborated clearly.

In this work, I will try to include all important steps of ML modeling (even though some are not necessary for this dataset) to make a consistent and tangible example, especially for geoscientists. Eight important ML algorithms will be examined and results will be compared. I will try to have an argumentative model evaluation discussion. I will not go deep into the algorithm’s fundamentals.

The dataset ([facies_vectors.csv](https://github.com/KDSmel/mini_projects/blob/main/Practical_ML_Tutorial_Facies/facies_vectors.csv)) for this study comes from Hugoton and Panoma Fields in North America. It consists of log data(the measurement of physical properties of rocks) of nine wells. We will use these log data to train supervised classifiers in order to predict discrete facies groups. For more detail, you may take a look here. The seven features are:

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

To access the dataset and jupyter notebook find out my [Git](https://github.com/KDSmel/mini_projects/tree/main/Practical_ML_Tutorial_Facies).


**This tutorial has four parts:**

Part.1: Exploratory Data Analysis,

Part.2: Build Model & Validate,

Part.3: Model Evaluation-1,

Part.4: Model Evaluation-2



**1. Exploratory Data Analysis**

1-1. Data visualization
      
	1–1–1. log-plot
	      
	1–1–2. Bar plot
	      
	1–1–3. Cross-plot
      
1–2. Feature Engineering

1–2–1. NaN imputation

1–2–2. Feature extraction

1–2–3. Oversampling

1–3. Feature Importance

1–3–1. Feature linear correlation

1–3–2. Decision tree

1–3–3. Permutation feature importance

**2. Build Model & Validate**

      2–1. Baseline Model
      
      2–2. Hyper-parameters
      
      2–3. Grid search

**3. Model Evaluation-1**

      3–1. Model metrics plot
      
      3–2. Confusion matrix

**4. Model Evaluation-2**

      4–1. Learning curves
      
      4–2. ROC plot
      
      4–3. Blind well prediction and evaluation
      

You can find the jupyter notebook file for this tutorial [here](https://github.com/KDSmel/mini_projects/blob/main/Practical_ML_Tutorial_Facies/Practical_Tutorial_ML_Facies.ipynb).

    
## Part.1: Exploratory Data Analysis
After data reading into python using Pandas, we can visualize it to understand data better. Before plotting, we need to define a color map(this step deserves to be in the Feature engineering part but we need here to plot color for facies classes) and devote color code for each facies.

Note1: codes embedded in this manuscript are presented to understand the work procedure. If you want to exercise by yourself, I highly recommend using the [jupyter notebook file](https://github.com/KDSmel/mini_projects/blob/main/Practical_ML_Tutorial_Facies/Practical_Tutorial_ML_Facies.ipynb).

Note2: shuffling data can cause differences between your runs and what appears here.


**1–1 Data visualization**
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

```python
cn = Counter(data.FaciesLabels)
for i,j in cn.items():
    percent = j / len(data) * 100
    print('Class=%s, Count=%d, Percentage=%.3f%%' % (i, j, percent))
# Class=FSiS, Count=663, Percentage=17.919%
# Class=CSiS, Count=851, Percentage=23.000%
# Class=PS, Count=646, Percentage=17.459%
# Class=WS, Count=511, Percentage=13.811%
# Class=D, Count=124, Percentage=3.351%
# Class=SiSh, Count=264, Percentage=7.135%
# Class=MS, Count=277, Percentage=7.486%
# Class=BS, Count=185, Percentage=5.000%
# Class=SS, Count=179, Percentage=4.838%

plt.bar(cn.keys(), cn.values(), color=facies_colors )
plt.title('Facies Distribution')
plt.ylabel('Frequency')
```

![image](./img2.png)

This is an imbalanced dataset. Dolomite has the lowest member participation. Comparing coarse siltstone, dolomite appears 8 times less than that.

1–1–3 Cross plot

To visualize multiple pairwise bivariate distributions in a dataset, we may use the pairplot() function from the seaborn library. It shows the relationship for the combination of variables in the dataset in the matrix format with a univariate distribution plot in diagonal. It is clear that PE log has a non-linear relationship with average porosity. Other pairs do not show a clear pattern. The distribution pattern in diagonal shows that each label class (facies) with respect to each feature has acceptable separation although there is a strong overlap for various classes. The ideal pattern can be assumed as a clear separation of distribution plots in tall bell shape normal distribution graph.

```python
sns_plot = sns.pairplot(data.drop(['Well Name','Facies','Formation','Depth','NM_M','RELPOS'],
                                  axis=1),
             hue='FaciesLabels', palette=facies_color_map,
             hue_order=list(reversed(facies_labels)))
sns_plot.savefig('cross_plots.png')
```

![image](./img3.png)

**Highlight:** Collinear features are features that are highly correlated with each other. In machine learning, these lead to decreased generalization performance on the test set due to high variance and less model interpretability. In this dataset, we are not facing with collinearity. Using data.corr() command:

![image](./tab1.png)


**1–2 Feature Engineering**

   1–2–1 NaN imputation

It is common to have missing value in the dataset. To see the sum of null values for each column of features:

```python
DataFrame.isna().sum()
# to find out which wells do not have PE
df_null = data_fe.loc[data_fe.PE.isna()]
df_null['Well Name'].unique()
#Categories (3, object): [ALEXANDER D, KIMZEY A, Recruit F9]
```

![image](./img4.png)

Here, PE has 917 null values.

There are several ways to deal with Null values in the dataset. The simplest approach is to drop the rows containing at least one null value. This can be logical with a bigger size dataset but in small data frames, single points are important. We can impute null values with mean or from adjacent data points in columns. Filling with mean value will not affect data variance and therefore will not have an impact on prediction accuracy, though can create data bias. Filling with the neighbor cells of column values can be appropriate if we have a geologically homogeneous medium like mass pure carbonate rocks.

Another approach, that I will implement here, to employe machine learning models to predict missing values. This is the best way of dealing with this dataset because we have just a single feature missing from the dataset, PE. On the other hand, filling with ML prediction is much better than the single mean value because we are able to see ML correlation and accuracy by dividing data to train and test sets.

Here, I will employ the Multi-Layer Perceptron Neural Network from scikit-learn to predict target value. I am not going to deep for this approach and use simply to predict missing values.

```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# select features and target log that has value

set_PE = data_fe[['Facies','Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']].dropna()  
X = set_PE[['Facies','Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]  # feature selection without null value
XX = data_fe[['Facies','Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
y = set_PE['PE'] # target log

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_b = scaler.fit_transform(XX)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MLP_pe = MLPRegressor(random_state=1, max_iter= 500).fit(X_train, y_train) #fit the model
MLP_pe.score(X_test, y_test) # examine accuracy
# accuracy: 0.7885539544025157

data_fe['PE_pred'] = MLP_pe.predict(X_b)  # predict PE
data_fe.PE.fillna(data_fe.PE_pred, inplace =True) # fill NaN vakues with predicted PE

```

![image](./img5.png)

Predicted PE in well ALEXANDER D shows the normal range and variation. Prediction accuracy is 77%.

1–2–2 Feature Extraction

Having a limited set of features in this dataset can lead us to think about extracting some data from the existing dataset. First, we can convert the formation categorical data into numeric data. Our background knowledge can help us to guess that some facies are possibly present more in a specific formation rather than others. We can use the LabelEncoder function:

```python
data_fe[‘Formation_num’] = LabelEncoder().fit_transform(data_fe[‘Formation’].astype(‘str’)) + 1
```

We converted formation category data into numeric to use as a predictor and added 1 to start predictor from 1 instead of zero. To see if new feature extraction would assist prediction improvement, we should define a baseline model then compare it with the extracted feature model.

**Baseline Model Performance**

For simplicity, we will use a logistic regression classifier as a baseline model and will examine model performance with a cross-validation concept. Data will be split into 10 subgroups and the process will be repeated 3 times.

```python
from numpy import mean
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X = data_fe[['Depth', 'GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'Formation_num']]
y = data_fe['Facies']

model = LogisticRegression(solver='liblinear')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f' % (mean(scores)))

#Accuracy: 0.561
```

Here, we can explore whether feature extraction can improve model performance. There are many approaches while we will use some transforms for chaining the distribution of the input variables such as Quantile Transformer and KBins Discretizer. Then, will remove linear dependencies between the input variables using PCA and TruncatedSVD. To study more refer [here](https://machinelearningmastery.com/quantile-transforms-for-machine-learning/).

Using feature union class, we will define a list of transforms to perform results aggregated together. This will create a dataset with lots of feature columns while we need to reduce dimensionality to faster and better performance. Finally, Recursive Feature Elimination, or RFE, the technique can be used to select the most relevant features. We select 30 features.

```python
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
#-------------------------------------------------- append transforms into a list
transforms = list()
transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('pca', PCA(n_components=7)))
transforms.append(('svd', TruncatedSVD(n_components=7)))
#-------------------------------------------------- initialize the feature union
fu = FeatureUnion(transforms)
#-------------------------------------------------- define the feature selection
rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=30)
#-------------------------------------------------- define the model
model = LogisticRegression(solver='liblinear')
#-------------------------------------------------- use pipeline to chain operation
steps = list()
steps.append(('fu', fu))
steps.append(('rfe', rfe))
steps.append(('ml', model))
pipeline = Pipeline(steps=steps)
# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f' % (mean(scores)))

# Accuracy: 0.605
```

Accuracy improvement shows that feature extraction can be a useful approach when we are dealing with limited features in the dataset.

1–2–3 Oversampling

In imbalanced datasets, we can use the resampling technique to add some more data points to increase members of minority groups. This can be helpful whenever minority label targets have special importance such as credit card fraud detection. In that example, fraud can happen with less than 0.1 percent of transactions while it is important to detect fraud.

In this work, we will add pseudo observation for the Dolomite class which has the lowest population

**Synthetic Minority Oversampling Technique, SMOTE:** the technique is used to select nearest neighbors in the feature space, separate examples by adding a line, and producing new examples along the line. The method is not merely generating the duplicates from the outnumbered class but applied K-nearest neighbors to generate synthetic data.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()

X_sm , y_sm = smote.fit_sample(X,y)
print("Before SMOTE: ", Counter(y))
print("After SMOTE: ", Counter(y_sm))
# Before SMOTE:  Counter({2: 851, 3: 663, 8: 646, 6: 511, 5: 277, 4: 264, 9: 185, 1: 179, 7: 124})
# After SMOTE:  Counter({3: 851, 2: 851, 8: 851, 6: 851, 7: 851, 4: 851, 5: 851, 9: 851, 1: 851})

model_bal = LogisticRegression(solver='liblinear')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_bal, X_sm, y_sm, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f' % (mean(scores)))

#Accuracy: 0.605
```

Accuracy improved by 3 percent but in multi-class classification, accuracy is not the best evaluation metric. We will cover others in the part.3.

**1–3 Feature Importance**

Some machine learning algorithms (not all) offer an importance score to help the user to select the most efficient features for prediction.

1–3–1 Feature linear correlation

The concept is simple: features have a higher correlation coefficient with target values are important for prediction. We can extract these coef’s like:

```python

# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define dataset
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('Logistic Regression Coefficients as Feature Importance Scores')
pyplot.show()
# Feature: 0, Score: 0.00099
# Feature: 1, Score: -0.00662
# Feature: 2, Score: -0.62498
# Feature: 3, Score: -0.04762
# Feature: 4, Score: 0.04542
# Feature: 5, Score: 0.95451
# Feature: 6, Score: 3.24906
# Feature: 7, Score: 0.36231
# Feature: 8, Score: -0.01491
```

![image](./img6.png)

1–3–2 Decision tree

This algorithm provides importance scores based on the reduction in the criterion used to split in each node such as entropy or Gini.

```python

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('Decision tree classifier Feature Importance Scores')
pyplot.show()

#Feature: 0, Score: 0.17175
#Feature: 1, Score: 0.10347
#Feature: 2, Score: 0.10918
#Feature: 3, Score: 0.08027
#Feature: 4, Score: 0.09843
#Feature: 5, Score: 0.10003
#Feature: 6, Score: 0.17501
#Feature: 7, Score: 0.11437
#Feature: 8, Score: 0.04749
```

![image](./img7.png)

1–3–3 Permutation feature importance

[Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) is a model inspection technique that can be used for any fitted estimator when the data is tabular. This is especially useful for non-linear or opaque estimators. The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled.

```python

from sklearn.inspection import permutation_importance
model = LogisticRegression(solver='liblinear')
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('Permutation Feature Importance Scores')
pyplot.show()

#Feature: 0, Score: -0.00130
#Feature: 1, Score: 0.01514
#Feature: 2, Score: 0.03400
#Feature: 3, Score: 0.04205
#Feature: 4, Score: 0.08692
#Feature: 5, Score: 0.07124
#Feature: 6, Score: 0.27622
#Feature: 7, Score: 0.03832
#Feature: 8, Score: 0.01811
```

![image](./img8.png)

In all these feature importance plots we can see that predictor number 6 (PE log) has the most importance in label prediction. Based on the model that we select to evaluate the result, we may choose features based on their importance and neglect the rest to speed up the training process. This is very common if we are rich in feature quantity, though in our example dataset here, we will use all features as predictors are limited.

**Summary**

Data preparation is one of the most important and time-consuming steps in machine learning. Data visualization can help us to understand data nature, borders, and distribution. Feature engineering is required especially if we have null and categorical values. In small datasets, feature extraction and oversampling can be helpful for model performances. Finally, we can analyze features in the dataset to see the importance of features for different model algorithms.

## Part.2: Build Model & Validate
In this part, we will build different models, validate them, and use the grid search approach to find out the optimum hyperparameters. 

The concept of model building in ML projects of scikit-learn libraries is simple. First, you select what type of model you are comfortable with, second, fit the model to the data using target and predictors(features), and finally, predict the unknown labels using available feature data.

In this project, we will use these classifiers to fit the feature data and then predict the facies classes. Here, we will not go into these algorithms' basic concepts. You may study on the scikit-learn website.

1. Logistic Regression Classifier
2. K Neighbors Classifier
3. Decision Tree Classifier
4. Random Forest Classifier
5. Support Vector Classifier
6. Gaussian Naive Bayes Classifier
7. Gradient Boosting Classifier
8. Extra Tree Classifier

**2–1 Baseline Model**

The philosophy of constructing a baseline model is simple: we need a basic and simple model to see how the adjustments on both data and model parameters can cause improvement in model performance. In fact, this is like a scale for comparison.

In this code script, we first defined our favorite model classifiers. Then, established baseline_model function. In this function, we employed the Pipeline function to implement step wised operation of data standard scaling(facilitate model running more efficient) and model object calling for cross-validation. I like Pipeline because it makes the codes more tidy and readable.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# define Classifiers
log = LogisticRegression()
knn = KNeighborsClassifier()
dtree = DecisionTreeClassifier()
rtree = RandomForestClassifier()
svm = SVC()
nb = GaussianNB()
gbc = GradientBoostingClassifier()
etree = ExtraTreesClassifier()

# define a function that uses pipeline to impelement data transformation and fit with model then cross validate
def baseline_model(model_name):

    model = model_name
    steps = list()
    steps.append(('ss', StandardScaler() ))
    steps.append(('ml', model))
    pipeline = Pipeline(steps=steps)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
     # balanced X,y from SMOTE can also be used 
    scores = cross_val_score(pipeline, X_sm, y_sm, scoring='accuracy', cv=cv, n_jobs=-1)

    print(model,'Accuracy: %.3f' % (mean(scores)))
 
#Run Function
baseline_model(log)
baseline_model(knn)
baseline_model(dtree)
baseline_model(rtree)
baseline_model(svm)
baseline_model(nb)
baseline_model(gbc)
baseline_model(etree)

#LogisticRegression() Accuracy: 0.623
#KNeighborsClassifier() Accuracy: 0.880
#DecisionTreeClassifier() Accuracy: 0.845
#RandomForestClassifier() Accuracy: 0.910
#SVC() Accuracy: 0.777
#GaussianNB() Accuracy: 0.357
#GradientBoostingClassifier() Accuracy: 0.832
#ExtraTreesClassifier() Accuracy: 0.934

```

[Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#:~:text=Cross%2Dvalidation%2C%20sometimes%20called%20rotation,to%20an%20independent%20data%20set.) sometimes called rotation estimation or out-of-sample testing is any of the various similar model validation techniques for assessing how the results of a statistical analysis will generalize to an independent data set. Models usually are overfitting when the accuracy score on training data is much higher than testing data. One way to examine model performance is to keep randomly some part of the dataset hold-out. This can be a weakness for small datasets. Another way is to divide the dataset into splits and run it while each split has a different set of test folds like the picture below. In this approach, cross-validation, models can examine all data without overfitting. However, for this project, we will keep a single well as a hold-out to examine model performances after all optimizations.

![image](./img21.png)
picture from scikit-learn

We repeated 3 times each operation over a dataset that already divided into 10 equal parts(folds) in cross-validation. In fact, we intended to expose the model to all data with a different combination of train and test sets without overlap.

Here, we used an average of accuracy as metrics for comparison of various model performances(accuracy and other evaluation metrics will be elaborated in the next post). It is used for simplicity, while for multi-class classification problems, accuracy is the weakest model evaluation approach. We will cover model evaluation metrics for multi-class classification problems in the next posts.

The Extra tree and Random forest classifier showed the best accuracy score for the facies label prediction while the Gaussian Naive Bayes classifier performed poorly.

**2–2 Hyper-parameters**

In machine learning, model parameters can be divided into two main categories:
**A- Trainable parameters:** such as weights in neural networks learned by training algorithms and the user does not interfere in the process,

**B- Hyper-parameters:** users can set them before training operations such as learning rate or the number of dense layers in the model.

Selecting the best hyper-parameters can be a tedious task if you try it by hand and it is almost impossible to find the best ones if you are dealing with more than two parameters.

One way is a random search approach. In fact, instead of using organized parameter searching, it will go through a random combination of parameters and look for the optimized ones. You may estimate that chance of success decreases to zero for larger hyper-parameter tuning.

Another way is to use skopt which is a scikit-learn library and uses Bayesian optimization that constructs another model of search-space for parameters. Gaussian Process is one kind of these models. This generates an estimate of how model performance varies with hyper-parameter changes. If you are interested in this approach, I have done an example for this data set, here. I will not repeat this process here because it is almost a big project.

**2–2–1 Grid Search**

This approach is to divide each parameter into a valid evenly range and then simply ask the computer to loop for the combination of parameters and calculate the results. The method is called Grid Search. Although it is done by machine, it will be a time-consuming process. Suppose you have 3 hyper-parameters with 10 possible values in each. In this approach, you will run 1⁰³ models (even with a reasonable training datasets size, this task is big).

You may see in the block code below that Gaussian Naive Bays is not involved because this classifier does not have a hyper-parameter that can strongly affect the model performance.

```python
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

#----------------------------------------------------------------logistic regression classifier
#define hyper parameters and ranges
param_grid_log = [{'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 
                   'max_iter':[100, 300]}]
#apply gridsearch
grid_log  = GridSearchCV(log, param_grid=param_grid_log, cv=5)
#fit model with grid search
grid_log.fit(X_train, y_train)
print('The best parameters for log classifier: ', grid_log.best_params_)

#----------------------------------------------------------------kNN classifier
#define hyper parameters and ranges
#define hyper parameters and ranges
param_grid_knn = [{'n_neighbors': [2, 3, 4, 6, 8, 10], 'weights': [ 'uniform', 'distance'], 
                   'metric': ['euclidean', 'manhattan', 'minkowski']}]
#apply gridsearch
grid_knn  = GridSearchCV(knn, param_grid=param_grid_knn, cv=5)
#fit model with grid search
grid_knn.fit(X_train, y_train)
print('The best parameters for knn classifier: ', grid_knn.best_params_)

#--------------------------------------------------------------decision tree classifier
#define hyper parameters and ranges
param_grid_dtree = [{'max_depth': [ 15, 20, 25, 30], 'criterion': ['gini',  'entropy']}]
#apply gridsearch
grid_dtree  = GridSearchCV(dtree, param_grid=param_grid_dtree, cv=5)
#fit model with grid search
grid_dtree.fit(X_train, y_train)
print('The best parameters for dtree classifier: ', grid_dtree.best_params_)

#--------------------------------------------------------------random forest classifier
#define hyper parameters and ranges
param_grid_rtree = [{'max_depth': [5, 10, 15, 20], 'n_estimators':[100,300,500] ,
                     'criterion': ['gini',  'entropy']}]
#apply gridsearch
grid_rtree  = GridSearchCV(rtree, param_grid=param_grid_rtree, cv=5)
#fit model with grid search
grid_rtree.fit(X_train, y_train)
print('The best parameters for rtree classifier: ', grid_rtree.best_params_)

#----------------------------------------------------------------SVM classifier
#define hyper parameters and ranges
param_grid_svm = [{'C': [100, 50, 10, 1.0, 0.1, 0.01], 'gamma': ['scale'], 
                   'kernel': ['poly', 'rbf', 'sigmoid'] }]
#apply gridsearch
grid_svm  = GridSearchCV(svm, param_grid=param_grid_svm, cv=5)
#fit model with grid search
grid_svm.fit(X_train, y_train)
print('The best parameters for svm classifier: ', grid_svm.best_params_)

#-----------------------------------------------------------------gbc classifier
#define hyper parameters and ranges
param_grid_gbc = [{'learning_rate': [0.1, 1], 'n_estimators':[200,350,500]}]
#apply gridsearch
grid_gbc  = GridSearchCV(gbc, param_grid=param_grid_gbc, cv=5)
#fit model with grid search
grid_gbc.fit(X_train, y_train)
print('The best parameters for gbc classifier: ', grid_gbc.best_params_)


#--------------------------------------------------------------extra tree classifier
#etree classifier
#define hyper parameters and ranges
param_grid_etree = [{'max_depth': [15, 20, 25, 30, 35], 'n_estimators':[200,350,500] , 
                     'criterion': ['gini',  'entropy']}]
#apply gridsearch
grid_etree  = GridSearchCV(etree, param_grid=param_grid_etree, cv=5)
#fit model with grid search
grid_etree.fit(X_train, y_train)
print('The best parameters for etree classifier: ', grid_etree.best_params_)

# The best parameters for log classifier:  {'C': 10, 'max_iter': 200, 'solver': 'lbfgs'}
# The best parameters for knn classifier:  {'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'distance'}
# The best parameters for dtree classifier:  {'criterion': 'entropy', 'max_depth': 20}
# The best parameters for rtree classifier: {'criterion': 'entropy', 'max_depth': 20, 'n_estimators': 500}
# The best parameters for svm classifier:  {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
#The best parameters for gbc classifier:  {'learning_rate': 0.1, 'n_estimators': 500}
#The best parameters for extra tree classifier: {'criterion': 'gini', 'max_depth': 35, 'n_estimators': 350}

```


The run time for the code above is almost an hour(depending on your computer configuration). I did not expand the grid search size for operation time-saving. You may extend for more than 2 by 2 parameters to search but expect that running time will be exponentially increased.

From line 69 in the code, you can see the best hyper-parameters that have been chosen for each classifier. Using these parameters, we can build these models with optimum hyper-parameters and run for the accuracy test.

![image](tab21.png)

Comparing with baseline model performances(results of first block codes in this post), hyper-parameters adjustment could help model performance to improve. It seems that for ensemble models hyper-parameters are not as efficient as the rests. Some drawbacks in the hyper-parameterized models comparing to baseline mean that we did not properly design the search range.

**Summary:**

In this part, we constructed eight models with default parameters, ran cross-validation. Then, we looked for hyper-parameters using the grid search approach. The best hyper-parameters are employed to build the model again and comparing the basic model we can see improvement in model performances.

## Part.3: Model Evaluation-1
In this part, we will elaborate on some model evaluation metrics specifically for multi-class classification problems. Accuracy, precision, recall, and confusion matrix are discussed below for our facies problem. 

When I was fresh in machine learning, I always considered constructing a model as the most important step of the ML tasks, while now, I have another concept; model evaluation skill is the fundamental key to modeling success. We need to make sure that our model is working well with new data. On the other hand, we have to be able to interpret various evaluation metrics to understand our model’s strengths and weaknesses leading us to model improvement hints. As we are dealing with the multi-class problem in this tutorial, we will focus on related evaluation metrics, but before that, we need to get familiar with some definitions.

**3–1 Model Metrics**

When we are working with classification problems, we will have 4 kinds of possibility with model outcomes:

**A) True Positive(TP)** is the outcome of the model correctly predicts the positive class. In our dataset, a positive class is a label that we are looking for specifically for that label prediction. For example, if we are analyzing ‘Dolomite’ class prediction, TP is the number of truly predicted Dolomite samples of test data by the model.

**B) True Negative(TN)** is an outcome where the model correctly predicts the negative class. Negative class in our dataset for Dolomite prediction are those facies classes that truly predicted as not Dolomite(predicted as the rest of classes and truly were not Dolomite).

**C) False Positive(FP)** is an outcome where the model incorrectly predicts the positive class. In our dataset, all facies classes that incorrectly predicted as Dolomite when we are evaluating Dolomite class prediction.

**D) False Negative(FN)** is an outcome where the model incorrectly predicts negative class. Again for Dolomite prediction, FN is the prediction of Dolomite as non-Dolomite classes.

**1.Accuracy:** it is simply calculated as a fraction of correct predictions over the total number of predictions.

Accuracy = (TP+TN) / (TP+TN+FP+FN)

**2. Precision:** this metric answers this question: what proportion of positive predictions is totally correct?

Precision = TP / (TP+FP)
looking at the equation, we can see that if a model has zero False Positive prediction, the precision will be 1. Again, in Dolomite prediction, this index shows what proportion of predicted Dolomite is truly Dolomite (not other facies are classified as Dolomite).

**3. Recall:** recall answer this question: what proportion of actual positives is classified correctly?

Recall= TP / (TP+FN)
looking at the equation, we can see that if a model has zero False Negative prediction, the recall will be 1. In our example, recall shows the proportion of Dolomite class that correctly identified by the model.

Note: to evaluate the model efficiency, we need to consider both precision and recall together. Unfortunately, these two parameters act against each other, improving one leads to decreasing the other. The ideal case is that both of them show near 1 values.

**4. f1_score:** The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and the worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

F1 = 2 * (precision * recall) / (precision + recall)

Let’s see one example of Logistic Regression classifier performance:

Run:
```python
from sklearn.metrics import precision_recall_fscore_support
model_log=LogisticRegression(C = 10, solver = ‘lbfgs’, max_iter= 200 ) 
model_log.fit(X_train, y_train)
y_pred_log = model_log.predict(X_test)
print(classification_report(y_test, y_pred_log, target_names= facies_labels))
```
![image](tab31.png)

To evaluate the Logistic Regression classifier performance, let's look at the first facies class Sandstone(SS). When this model predicts a facies as SS, it is correct in 75% of the time(Precision). On the other hand, this model correctly identifies 89% of all SS facies members(Recall). We can guess that f1_score is somewhere between these two metrics. Support means the individual class members for the test.

Let's have some block of codes to implement the above-mentioned procedure in order for all models and plot the result as an average. Up to line 15, we defined the model objects with hyper-parameters that we already obtained from the grid-search approach. Then(line 16 to 25) models are appended into a list to be iterable when we want to fit and cross-validate in order. After cross-validation, we stored metrics results in the list for each model. line 37 to 52, we established a for loop to calculate the average value of each of these metrics for each model. The rest of the code is a plotting task.

```python
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support

# define Classifiers with optimum hyper_param
log = LogisticRegression(C = 10, solver = 'lbfgs', max_iter= 200 ) 
knn = KNeighborsClassifier(leaf_size = 10, n_neighbors=2)
dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth=15)
rtree = RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=300)
svm = SVC(C=100, gamma=0.001)
nb = GaussianNB()
gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
etree = ExtraTreesClassifier(criterion='gini', max_depth=35, n_estimators=500)

# append models and run in for loop for cv
models = []
models.append(('log' , log))
models.append(('knn', knn))
models.append(('dtree', dtree))
models.append(('rtree', rtree))
models.append(('svm', svm))
models.append(('nb',  nb))
models.append(('gbc', gbc))
models.append(('etree', etree))
results = []
names = []
scoring =  ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
for name, model in models:
    cv = KFold(n_splits=10, shuffle=True , random_state=42)
    cv_results = cross_validate(model, X_sm, y_sm, cv=cv, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    results.append(name)
results =  results[::2] # use the numeric part

#average of each metrics
test_acc = []
for i in range (len(names)):
    test_acc.append(results[i]['test_accuracy'].mean())
    
test_f1 = []
for i in range (len(names)):
    test_f1.append(results[i]['test_f1_macro'].mean())  
    
test_pre = []
for i in range (len(names)):
    test_pre.append(results[i]['test_precision_macro'].mean())

test_rec = []
for i in range (len(names)):
    test_rec.append(results[i]['test_recall_macro'].mean())
    
 #Model Merices plot
category_names = names
result_data = {"accuracy_score": test_acc, 'f1score':    test_f1, 'precision': test_pre, 'recall':    test_rec,  }

def survey(result_data, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(result_data.keys())
    data = np.array(list(result_data.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0, 1, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
   

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.7,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color =  'k'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(np.around(c,decimals = 2)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, -0.22),
              loc='lower left', fontsize='large')
    ax.set_title('Diffrent Metrics Average', loc='center')
    fig.savefig('fname_macro', dpi=300)
    return fig, ax


survey(result_data, category_names)
plt.show()

```

![image](img31.png)

This is the plot that shows the average value of each of the evaluation metrics(y-axis) for individual employed models. Here, we wanted to compare all models performances overall. It seems that Extra tree and Random forest did the best prediction while Gaussian Naive Bays was not that much efficient predictor model.

If we are concerned about an individual facies prediction, we should consider eliminating the rest of the metrics from the ‘results’ list and run the program again.

**3–2 Confusion matrix**

The confusion matrix shows predicted class labels against original true label data. This is a fantastic visualization tool that we can see each class of facies is predicted correctly or wrong into other classes.

In the line of codes below, we first defined a function to make fancy use of the confusion matrix function developed by sklearn. After function definition, we fit and run the Logistic Regression classifier. Now we have predicted facies labels with true facies labels for test data. Calling the function with the required input parameters will create a plot of the confusion matrix.

```python
from sklearn.metrics import confusion_matrix
import itertools

# define function to implement confusion matrix with normalization capability
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# create Logistic Reg model with optimom hyper-parameters
model_log=LogisticRegression(C = 10, solver = 'lbfgs', max_iter= 500 ) 
model_log.fit(X_train, y_train)
# predict test labels
y_pred_log = model_log.predict(X_test)
#calculate confusion matrix
cnf_rtree = confusion_matrix(y_test, y_pred_log)

#Plot confusion matrix for RandomForest classifier
fig = plt.figure()
fig.set_size_inches(7, 6, forward=True)
plot_confusion_matrix(cnf_rtree, classes=np.asarray(facies_labels),
                      title='confusion matrix (Logistic Regression classifier)')

```

![image](img32.png)

Taking a look at the plot(first row), we recognize that this algorithm could predict 151 SS class correctly while 18 true SS were classified as CSiS incorrectly. From the previous section, we are familiar with the recall concept. From all true class members of SS(169), the classifier could recognize 151 correctly; 151/169 is 89%(we have seen this number in the class report in the picture above). So, we can conclude that if we move our evaluation in the row direction(True labels) we are dealing with recall.
You may guess that if we go in the column direction, we will deal with Precision. For SS precision is 75% as 149/201.

In the picture below, we see how the Naive Bayes classifier poorly performed prediction. This classifier is totally overestimated BS class in prediction.

![image](img33.png)


Up to now, we have some metrics that helped us to evaluate the model performances but still, we can not guarantee that which one is the best. Because some models can memorize training data and follow data complexity severely and when it faces a new dataset, its performance will be poor. This is called over-fitting (model with high variance). A model with high variance will change a lot with small changes to the training dataset.
On the other hand, when a model too generalizes prediction, it will not be able to capture the complexity of a dataset, this is called under-fitting(model with high bias).
Our ideal model is something between these two models leaving us for bias-variance trade-off.

The question is: how we can recognize that our model is over-fit or under-fit?
We will cover in the next part of this tutorial.

**Summary:**

Model evaluation is the most important task in ML model production. We mainly start with simple evaluation metrics and then narrow down to specific and more detailed metrics to understand our model's strengths and weaknesses.

## Part.4: Model Evaluation-2
In this part, we will elaborate on more model evaluation metrics specifically for multi-class classification problems. Learning curves will be discussed as a tool to come up with an idea of how to trade-off between bias and variance in the model parameter selection. ROC curves for all classes in a specific model will be shown to see how false and true positive rate varies through the modeling process. Finally, we will select the best model and examine its performance on blind well data(data that was not involved in any of the processes up to now). 

![image](img41.png)

**4–1 Learning Curves**
Let’s look at the schematic graph of the validation curve. As the model complexity increases, the training score increases as well but at some point, the validation(or test data) score starts to decrease. Increasing model complexity will lead to high variance or over-fitting. When the model is too simple, it can not capture all aspects of data mapping complexity leaving a high bias model. The best model is located between these two conditions, where it is complicated enough to have the highest validation score while not too complicated to capture every detail of training data.

![image](img42.png)
figure from: jakevdp.github.io

Let’s create a function to plot the learning curve for our dataset. This function will generate 8 plots (each model algorithms) for test and training learning curve, samples vs fit time curve, and fit the time vs score curve. The function will receive these parameters: model estimator, the title for the chart, axes location for each model, ylim, cross-validation value, number of jobs that can be done in parallel, and train data size.

```python
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

fig, axes = plt.subplots(6, 4, figsize=(15, 30))
cv = 5
# use models tuple which already made with optimized hyper-parameters
title = "Learning Curves (Logistic Reg.)"
plot_learning_curve(models[0][1], title,  X_sm, y_sm, axes=axes[0:3, 0], ylim=(0.1, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves(KNN)"
plot_learning_curve(models[1][1], title,  X_sm, y_sm, axes=axes[0:3, 1], ylim=(0.1, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves(Decision tree)"
plot_learning_curve(models[2][1], title,  X_sm, y_sm, axes=axes[0:3, 2], ylim=(0.1, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves(Random Forest)"
plot_learning_curve(models[3][1], title,  X_sm, y_sm, axes=axes[0:3, 3], ylim=(0.1, 1.01),  cv=5, n_jobs=4)

title = "Learning Curves(Support Vector)"
plot_learning_curve(models[4][1], title, X_sm, y_sm, axes=axes[3:7, 0], ylim=(0.1, 1.01),  cv=cv, n_jobs=4)

title = "Learning Curves(Naive Bayes)"
plot_learning_curve(models[5][1], title, X_sm, y_sm, axes=axes[3:7, 1], ylim=(0.1, 1.01),  cv=cv, n_jobs=4)

title = "Learning Curves(Gradient Boosting)"
plot_learning_curve(models[6][1], title, X_sm, y_sm, axes=axes[3:7, 2], ylim=(0.1, 1.01),   cv=cv, n_jobs=4)

title = "Learning Curves(Extra Tree)"
plot_learning_curve(models[7][1], title, X_sm, y_sm, axes=axes[3:7, 3], ylim=(0.1, 1.01),   cv=cv, n_jobs=4)
plt.savefig('Models10003.png', dpi=300)
plt.show()

```

If you refer to the code above, you may notice that this figure consists of two sets of rows of graphs. The first row in each set belongs to the learning curve of the first four models, then in the second row, fitting time is plotted as a function of training sample sizes and in the third row, the score is plotted as the function of fitting time. The second set of rows is the same as above but for different models.

![image](img43.png)

There are some important points to consider:
1. For all algorithms, you may notice that training scores are always higher than tests or cross-validation scores(this is almost standard of ML).
2. For logistic regression, SVM, and Naive Bays we see a specific pattern. The Traning data score decreases as examples increase in the training dataset and the validation score is very low(high bias) at the beginning and increases. This pattern can be found in more complex datasets very often.
3. For the rest of the classifiers, we can see that the training score is still around the maximum, and validation could be increased with more new data samples. Comparing with the schematic chart above, we do not see the turning point for the validation curve in these plots meaning we are not in the area of over-fitting at the end of the training. We also can not claim that these classifiers(such as the random forest algorithm) have the highest performance because non of the validation curves did not flatten at the end of the training process.

The plots in the second row show the times required by the models to train with various sizes of training datasets. The plots in the third row show how much time was required to train the models for each training size.

**4–2 ROC Curves**

Receiver Operating Characteristic (ROC) is a metric to evaluate classifier output quality. To calculate and plot:

```python
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Binarize the output
yy = label_binarize(y_sm, classes=[ 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y = label_binarize(y, classes=np.asarray(facies_labels))
n_classes = yy.shape[1]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, yy, test_size=.3,  random_state=0)
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(log)
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


lw=1
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

colors = cycle(['b', 'y', 'g','r','k','c','m'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logestic Regression Classifier')
plt.legend(loc="lower right")
plt.savefig('ROC_log.png', dpi=300)
plt.show()

```

ROC curves typically consist of true positive rates on the y-axis and false positive rates on the x-axis. This means that the top left corner of the graph area is the ideal point as the true positive is maximum and the false positive is zero. As always not only we do not have a perfect dataset but also datasets are contaminated by some noise levels, which is not very realistic. However, it is always good to have a larger area under the curve(in the figure below, the first and last classes are the greatest). If we go back to the precision-recall report showed at the start of this post, we see that precision and recall(and f-score of course) for the first and last classes were the highest among others.

![image](img432.png)

ROC curves can be plotted for other model algorithms as well.

**4-3 Blind Well Prediction**
This is the final step of this project. We will see how the constructed models can predict with real data that already have not been seen during the training and test process. We have kept well ‘KIMZEY A’ as blind well out of the process because it has all facies class. In this set of codes below, first defined X and y array data for blind well, then copied optimized model objects from the previous section and fit the models with all available data expect blind well data. Then predict blind labels and stored accuracy score.
We defined a function to plot all prediction results juxtaposed by original true labels.

```python
X_blind = blind[['Depth', 'GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'Formation_num']]
y_blind = blind['Facies']
# scale
scaler = StandardScaler()
X_blind = scaler.fit_transform(X_blind)


# define Classifiers
log = LogisticRegression(C = 10, solver = 'lbfgs', max_iter= 300 ) 
knn = KNeighborsClassifier(leaf_size = 10, n_neighbors=2)
dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth=15)
rtree = RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=300)
svm = SVC(C=100, gamma=0.001)
nb = GaussianNB()
gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
etree = ExtraTreesClassifier(criterion='entropy', max_depth=50, n_estimators=500)


%%time
log.fit(X_sm,y_sm)
y_hat_log = log.predict(X_blind)
blind['log_Pred'] = y_hat_log   #add to df
acc_log = (accuracy_score(y_blind, y_hat_log ))

knn.fit(X_sm,y_sm)
y_hat_knn = knn.predict(X_blind)
blind['knn_Pred'] = y_hat_knn   #add to df
acc_knn = (accuracy_score(y_blind, y_hat_knn ))

dtree.fit(X_sm,y_sm)
y_hat_dtree = dtree.predict(X_blind)
blind['dtree_Pred'] = y_hat_dtree   #add to df
acc_dtree = (accuracy_score(y_blind, y_hat_dtree ))

rtree.fit(X_sm,y_sm)
y_hat_rtree = rtree.predict(X_blind)
blind['rtree_Pred'] = y_hat_rtree   #add to df
acc_rtree = (accuracy_score(y_blind, y_hat_rtree ))

svm.fit(X_sm,y_sm)
y_hat_svm = svm.predict(X_blind)
blind['svm_Pred'] = y_hat_svm   #add to df
acc_svm = (accuracy_score(y_blind, y_hat_svm ))

gbc.fit(X_sm,y_sm)
y_hat_gbc = gbc.predict(X_blind)
blind['gbc_Pred'] = y_hat_gbc   #add to df
acc_gbc = (accuracy_score(y_blind, y_hat_gbc ))

etree.fit(X_sm,y_sm)
y_hat_etree = etree.predict(X_blind)
blind['etree_Pred'] = y_hat_etree   #add to df
acc_etree = (accuracy_score(y_blind, y_hat_etree ))


#create dataframe to compare accuracy results
#create dataframe to compare accuracy results
d = { 'model':['log' , 'knn',  'dtree', 'rtree', 'svm', 'gbc', 'etree'],
      'test_score':[test_acc[0], test_acc[1],test_acc[2], test_acc[3],test_acc[4], test_acc[6], test_acc[7]],
      'blind_score': [acc_log, acc_knn, acc_dtree, acc_rtree, acc_svm, acc_gbc, acc_etree ]}

df_comp = pd.DataFrame(d) 

#Create a section to compare various model performance in facies prediction

def compare_all_facies(logs, Pred1, Pred2, Pred3, Pred4, Pred5, Pred6, Pred7, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[Pred1].values,1), 100, 1)
    cluster3 = np.repeat(np.expand_dims(logs[Pred2].values,1), 100, 1)
    cluster4 = np.repeat(np.expand_dims(logs[Pred3].values,1), 100, 1)
    cluster5 = np.repeat(np.expand_dims(logs[Pred4].values,1), 100, 1)
    cluster6 = np.repeat(np.expand_dims(logs[Pred5].values,1), 100, 1)
    cluster7 = np.repeat(np.expand_dims(logs[Pred6].values,1), 100, 1)
    cluster8 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

   
    f, ax = plt.subplots(nrows=1, ncols=8, figsize=(12, 6))

    im1 = ax[0].imshow(cluster1, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im2 = ax[1].imshow(cluster2, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im3 = ax[2].imshow(cluster3, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im4 = ax[3].imshow(cluster4, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im5 = ax[4].imshow(cluster5, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im6 = ax[5].imshow(cluster6, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im7 = ax[6].imshow(cluster7, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
    im8 = ax[7].imshow(cluster8, interpolation='none', aspect='auto',
                       cmap=cmap_facies,vmin=1,vmax=9)
       
    
    divider = make_axes_locatable(ax[7])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar=plt.colorbar(im8, cax=cax)
    cbar.set_label((5*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-8):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=2)
    
    ax[0].set_xlabel('True_Facies'); ax[1].set_xlabel(Pred1); ax[2].set_xlabel(Pred2)
    ax[3].set_xlabel(Pred3); ax[4].set_xlabel(Pred4); ax[5].set_xlabel(Pred5)
    ax[6].set_xlabel(Pred6); ax[7].set_xlabel('True_Facies')
    
    #ax[0].set_yticklabels([]) ;
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([]) 
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[6].set_yticklabels([])
    ax[7].set_yticklabels([])
    
    ax[0].set_xticklabels([]); ax[1].set_xticklabels([]); ax[2].set_xticklabels([])
    ax[3].set_xticklabels([]); ax[4].set_xticklabels([]); ax[5].set_xticklabels([])
    ax[6].set_xticklabels([]); ax[7].set_xticklabels([])

    f.suptitle('Various model predictions in well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    
compare_all_facies(blind,'log_Pred','knn_Pred','dtree_Pred', 'rtree_Pred','svm_Pred','gbc_Pred', 
                   'gbc_Pred','etree_Pred', facies_colors)
# plt.savefig("Compo.png", dpi=400)

```

The blind data model accuracy reveals that the best accuracy belongs to the extra tree classifier(ensemble in total). We see that although prediction accuracy for the training-test dataset is high for all models it decreases in blind well validation. We can conclude that the number of data samples and presumably amount of features are not high enough for models to capture all aspects of data complexity. In fact, more samples and more data features can help to improve model prediction accuracy.

![image](img44.png)

Classification report below for Extras Tree Classifier: Facies SS was failed in prediction while BS with 7 members could have been recognized by the extra tree model. Mudstone(MS) showed weak prediction results.

![image](tab41.png)

Confusion matrix, Extras Tree Classifier: confusion matrix shows us the extra tree model capability to predict facies labels. We see that MS true labels are predicted as WS and PS by the model.

![image](img45.png)

Finally, the true facies labels(left track) is plotted as a criterion to compare with different model prediction results in the blind well. Visually, extras tree and random forest model prediction seem better than the rest of the models. One important point that we can see is that classifiers try to detect thin layers inside thick layers.

![image](img46.png)

As a final point, selecting the best model can be a subjective topic depending on your expectancy from a model. For example, if you are looking for a specific thin layer in geological succession, and a model could predict that layer very well while did poor prediction for other layers, we can employ that model, though we are aware that its evaluation metrics are not good enough altogether.

**Conclusion:**

To build a machine learning model the first step is to prepare the dataset. Data visualization, feature engineering, and extracting important features can be an important part of data preparation. Null values management is also a very crucial step specifically in the small datasets. We imputed missing data in this dataset using ML prediction to save more data. The second main important step is to build a model and validate it. Hyper-parameters are also important to be chosen carefully for efficient model performances. We employed a grid search to find out the optimized parameters. Finally, model evaluation is the most important task in ML model production. We mainly start with simple evaluation metrics and then narrow down to specific and more detailed metrics to understand our model’s strengths and weaknesses.
