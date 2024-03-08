#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time

start_time = time.time()  # Start time



import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from pathlib import Path
from collections import defaultdict
from features.utils import build_mapping_to_ids

warnings.filterwarnings('ignore')


# # Data

# ### Get all problems

# In[2]:


dataset = pd.read_csv("data/java_test_dataset_code.csv")


# In[3]:


# Drop rows where null values exist in 'focal_class_code' or 'test_class_code'
dataset.dropna(subset=['focal_class_code', 'test_class_code'], inplace=True)


# In[4]:


dataset.head()


# In[5]:


dataset.info()


# In[6]:


dataset.isnull().sum()


# # Build dataset

# In[7]:


from features import *
from sklearn.feature_selection import mutual_info_regression


# In[8]:


# codes = dataset['focal_class_code'].values  # Assuming 'code_column' is the name of your column with the code

codes_with_ids = [{'repo_id': row['repo_id'], 'code': row['focal_class_code']} for index, row in dataset.iterrows()]

samples = calculate_features_for_files(codes_with_ids)


# ### Minor EDA for samples

# In[9]:


samplesdf = pd.DataFrame(samples) 


# In[10]:


samplesdf.shape


# In[11]:


column_name = 'repo_id'
# Pop the column out of the DataFrame
desired_column = samplesdf.pop(column_name)
# Reinsert it at the beginning of the DataFrame
samplesdf.insert(0, column_name, desired_column)


# In[12]:


samplesdf.head()


# In[13]:


# columns_to_check = [
#     "WordUnigramTF",
#     "In(numkeywords/length)",
#     "In(numTernary/length)",
#     "In(numTokens/length)",
#     "In(numComments/length)",
#     "In(numLiterals/length)",
#     "In(numKeywords/length)",
#     "In(numFunctions/length)",
#     "In(numMacros/length)",
#     "nestingDepth",
#     "branchingFactor",
#     "avgParams",
#     "stdDevNumParams",
#     "avgLineLength",
#     "stdDevLineLength",
#     "In(numTabs/length)",
#     "In(numSpaces/length)",
#     "In(numEmptyLines/length)",
#     "whiteSpaceRatio",
#     "newLineBeforeOpenBrace",
#     "tabsLeadLines",
#     "MaxDepthASTNode",
#     "ASTNodeBigramsTF",
#     "ASTNodeTypesTF",
#     "ASTNodeTypesTFIDF",
#     "ASTNodeTypeAvgDep",
#     "cppKeywords",
#     "CodeInASTLeavesTF",
#     "CodeInASTLeavesTFIDF",
#     "CodeInASTLeavesAvgDep"
# ]

# # Function to clean column names
# def clean_column_name(name):
#     return name.replace(" ", "").replace("In", "ln").lower()

# # Clean DataFrame column names
# xdf.columns = [clean_column_name(name) for name in xdf.columns]

# # Check each column
# for col in columns_to_check:
#     cleaned_col = clean_column_name(col)
#     if cleaned_col in xdf.columns:
#         print(f"Column '{col}' exists in the DataFrame.")
#     else:
#         print(f"Column '{col}' does NOT exist in the DataFrame.")


# ## Build X and Y 

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import numpy as np


# In[22]:


y = samplesdf['repo_id']
X = samplesdf.drop(columns=['repo_id'])


# In[23]:


X.fillna(0, inplace=True)


# ### Select the best 1500 features according to mutual information

# In[24]:


# Convert to numpy array for mutual information calculation if needed
X_np = X.to_numpy(dtype=np.float32)

# Replace inf/-inf with large finite numbers (if infinities are expected)
X_np = np.where(np.isinf(X_np), np.finfo(np.float32).max, X_np)


# In[25]:


# Calculate mutual information
mi = mutual_info_regression(X_np, y, random_state=0)
mi /= np.max(mi)  # Normalize mutual information scores for better comparison

# Select the top 1500 features
mi_indices = np.argsort(mi)[-1500:]  # Get indices of top 1500 features
selected_features = X.columns[mi_indices]  # Get feature names
X = X[selected_features]  # Subset X to keep only selected features

print(f'Number of samples: {X.shape[0]}')
print(f'Number of features: {X.shape[1]}')


# In[26]:


X.head()


# In[27]:


X.isnull().all(axis=1).sum() # Rows with all columns null


# In[28]:


X.isnull().sum()


# # Classification

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')


# # Validation

# ### Cross-Validation

# In[33]:


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)

print("CV Accuracy Scores:", cv_scores)
print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


# ### Precision, Recall, and F1 Score

# In[38]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### ROC Curve and AUC Score

# In[39]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# Binarize y_test and y_pred for multi-class ROC AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
y_pred_proba = rf.predict_proba(X_test)

# Compute AUC
roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr")

print("ROC AUC Score:", roc_auc)

# For plotting ROC curves for each class, you'd iterate through classes and calculate ROC curve per class


# ### Feature Importances

# In[40]:


feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')  # Top 10 features
plt.show()


end_time = time.time()  # End time
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")