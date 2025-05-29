# ML LEARNING JOURNEY: QUICK REFERENCE GUIDE

## COMMON IMPORTS
```python
# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
```

## DATASET LOADING
```python
# Sample datasets
titanic = pd.read_csv('data/sample/titanic_sample.csv')
housing = pd.read_csv('data/sample/housing_sample.csv')
iris = pd.read_csv('data/sample/iris_sample.csv')

# Scikit-learn built-in datasets
from sklearn.datasets import load_iris, load_boston, load_digits
X, y = load_iris(return_X_y=True)
```

## DATA PREPROCESSING
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling missing values
df.isnull().sum()  # Check missing values
df.fillna(df.mean())  # Fill with mean
df.dropna()  # Drop rows with missing values
```

## MODEL TRAINING & EVALUATION
```python
# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Classification
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## VISUALIZATION TEMPLATES
```python
# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='column_name', kde=True)
plt.title('Distribution of Column')
plt.savefig('visualizations/distribution.png')

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x_column', y='y_column', hue='category_column')
plt.title('Relationship between X and Y')
plt.savefig('visualizations/scatter.png')

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('visualizations/heatmap.png')
```

## MODEL PERSISTENCE
```python
# Save model
import pickle
with open('models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('models/my_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

## PYTORCH NEURAL NETWORK TEMPLATE
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Training loop
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## COMMON METRICS
- **Regression**: MSE, RMSE, MAE, R²
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC AUC
- **Clustering**: Silhouette Score, Davies-Bouldin Index

## M-V-A FRAMEWORK CHECKLIST
- **Model**: Is the implementation correct? Are parameters appropriate?
- **View**: Do visualizations clearly show patterns/results? Are axes labeled?
- **Analysis**: What worked well? What could be improved? What did you learn?

## DAILY WORKFLOW REMINDER
1. Morning: Review concepts, set up notebook structure
2. Afternoon: Implement model and visualizations
3. Evening: Analyze results, document in learning diary
4. Before finishing: Commit code, update progress

## USEFUL RESOURCES
- Documentation: [Python](https://docs.python.org/3/), [NumPy](https://numpy.org/doc/), [Pandas](https://pandas.pydata.org/docs/)
- ML: [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- DL: [PyTorch](https://pytorch.org/docs/stable/index.html)
- Viz: [Matplotlib](https://matplotlib.org/stable/contents.html), [Seaborn](https://seaborn.pydata.org/)
