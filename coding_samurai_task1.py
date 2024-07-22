#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[62]:


iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target


# In[63]:


# Step 2: Explore the Dataset
print(df.head())


# In[64]:


print(df.describe())


# In[65]:


print(df['species'].value_counts())


# In[66]:


print(df.tail)


# In[67]:


#data preprocesing
df.isnull().sum()


# In[68]:


import warnings
warnings.filterwarnings('ignore')

sns.pairplot(df, hue='species')
plt.show()


# In[70]:


# Plot the distribution of each feature
df.hist(figsize=(10, 8))
plt.suptitle('Feature Distributions')
plt.show()


# In[71]:


correlation_matrix = df.corr()
print(correlation_matrix)


# In[72]:


# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[73]:


# Step 3: Preprocess the Data

X = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[74]:


# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[75]:


models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}


# In[76]:


# Step 6: Train and Evaluate the Models
results = {}
for model_name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    results[model_name] = {
        "cv_scores": cv_scores,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }
    
    print(f"{model_name}:")
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")


# In[77]:


# Step 7: Hyperparameter Tuning for the Best Model
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Best Model:", best_model)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Best Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Best Model Classification Report:\n", classification_report(y_test, y_pred_best))


# In[78]:


# Step 8: Plot Learning Curves
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 50))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curves')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

plot_learning_curve(best_model, X_scaled, y)


# In[79]:


# Step 9: Save the Best Model
joblib.dump(best_model, 'iris_best_model.pkl')


# In[85]:


# Step 10: Load and Use the Saved Model
loaded_model = joblib.load('iris_best_model.pkl')
sample = [[2.0, 0.5, 0.4, 0.5]]  # Example data
sample_scaled = scaler.transform(sample)
print("Prediction for sample:", loaded_model.predict(sample_scaled))


# In[ ]:




