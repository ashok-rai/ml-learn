# 21-DAY INTENSIVE ML LEARNING ROADMAP

## Overview
This roadmap provides a structured, fast-paced learning journey through machine learning fundamentals and applications using the Model-View-Analysis (M-V-A) framework. Each day combines theory with hands-on practice to ensure rapid skill acquisition.

## How to Use This Roadmap
1. **Daily Kick-off (15 min)**: Review the day's topic and set specific goals
2. **Learning Blocks**: Complete 3 × 45-minute focused coding sessions
3. **End-of-day M-V-A Ritual (10 min)**:
   - **Model**: Document what you built
   - **View**: Save visualizations and results
   - **Analysis**: Write a brief critique and plan improvements
4. **Push your work**: Commit your progress and update your learning diary

## DAY 0: Setup (1-2 hours)
- [ ] Install Python 3.11, VS Code/Jupyter, create GitHub account
- [ ] Create repository "21-Day-ML-Sprint"
- [ ] Set up virtual environment and install requirements
- [ ] Create learning-diary.md for daily reflections

## FOUNDATIONS WEEK (Days 1-7)
*Goal: Build a solid Python + math foundation with daily mini-projects*

### Day 1: PYTHON CORE BLITZ
- [ ] **Model**: Write 5 functions (fizzbuzz, prime-check, fibonacci, etc.)
- [ ] **View**: Run doctests & simple print statements
- [ ] **Analysis**: Measure runtime with %timeit
- [ ] **Deliverable**: Push `core_python.ipynb`

### Day 2: NUMPY & PANDAS CRASH
- [ ] **Model**: Implement 3 NumPy array operations, 3 Pandas DataFrame operations on Titanic dataset
- [ ] **View**: Use `.head()`, create seaborn heatmap
- [ ] **Analysis**: Discuss missing-value strategy in diary
- [ ] **Deliverable**: Push `numpy_pandas.ipynb`

### Day 3: VISUALIZATION QUICK-FIRE
- [ ] **Model**: Create one example each with Matplotlib, Seaborn, and Plotly
- [ ] **View**: Present a mini "gallery" markdown with 3 images
- [ ] **Analysis**: Compare visualization libraries for different use cases
- [ ] **Deliverable**: Push `visualization_gallery.ipynb` and `gallery.md`

### Day 4: PROBABILITY & STATS PRIMER
- [ ] **Model**: Simulate coin flips, normal distribution sampling using NumPy
- [ ] **View**: Plot histograms; compute mean, variance, confidence intervals
- [ ] **Analysis**: Reflection on when to worry about p-values
- [ ] **Deliverable**: Push `probability_stats.ipynb`

### Day 5: LINEAR ALGEBRA LITE
- [ ] **Model**: Implement matrix multiplication two ways (loops vs np.dot)
- [ ] **View**: Visualize 2-D vectors and transformations
- [ ] **Analysis**: Benchmark performance and note observations on broadcasting
- [ ] **Deliverable**: Push `linear_algebra.ipynb`

### Day 6: CALCULUS FOR ML
- [ ] **Model**: Write & plot f(x)=x³; derive & plot derivative using autograd
- [ ] **View**: Visualize function and its gradient
- [ ] **Analysis**: Observe gradient shapes and document insights
- [ ] **Deliverable**: Push `calculus_ml.ipynb`

### Day 7: FIRST MINI-PROJECT
- [ ] **Model**: Build end-to-end linear regression on Boston Housing or California dataset
- [ ] **View**: Create scatter plot of predicted vs actual values
- [ ] **Analysis**: Calculate MAE, R², and document improvement ideas
- [ ] **Deliverable**: Push `mini_project_regression.ipynb`

## MODELING WEEK (Days 8-14)
*Goal: Explore breadth of classical ML models with daily deployable demos*

### Day 8: CLASSIFICATION BASICS
- [ ] **Model**: Implement logistic regression on breast cancer dataset
- [ ] **View**: Create confusion matrix heatmap + ROC curve
- [ ] **Analysis**: Discuss class imbalance and its impact
- [ ] **Deliverable**: Push `classification_basics.ipynb`

### Day 9: TREE-BASED METHODS
- [ ] **Model**: Implement DecisionTree & RandomForest on Titanic dataset
- [ ] **View**: Generate Graphviz tree plot; feature importance bar chart
- [ ] **Analysis**: Compare tree depth vs accuracy
- [ ] **Deliverable**: Push `tree_methods.ipynb`

### Day 10: BOOSTING DAY
- [ ] **Model**: Implement XGBoost/LightGBM quickstart
- [ ] **View**: Create SHAP summary plot
- [ ] **Analysis**: Discuss overfitting and strategies to prevent it
- [ ] **Deliverable**: Push `boosting_methods.ipynb`

### Day 11: SVM & MARGIN INTUITION
- [ ] **Model**: Implement SVM on 2-D toy dataset
- [ ] **View**: Plot decision boundary with hyperplane
- [ ] **Analysis**: Visual explanation of kernel trick (using RBF demo)
- [ ] **Deliverable**: Push `svm_intuition.ipynb`

### Day 12: UNSUPERVISED 1: K-MEANS & PCA
- [ ] **Model**: Apply K-means clustering and PCA to Iris dataset
- [ ] **View**: Create scatter plot colored by cluster
- [ ] **Analysis**: Document variance explained by principal components
- [ ] **Deliverable**: Push `kmeans_pca.ipynb`

### Day 13: UNSUPERVISED 2: TSNE & DBSCAN
- [ ] **Model**: Apply t-SNE to 1k samples from MNIST
- [ ] **View**: Generate density-based cluster visualization
- [ ] **Analysis**: Compare t-SNE with PCA for dimensionality reduction
- [ ] **Deliverable**: Push `tsne_dbscan.ipynb`

### Day 14: MODEL SELECTION & PIPELINES
- [ ] **Model**: Implement cross-validation and GridSearchCV pipeline on chosen dataset
- [ ] **View**: Plot validation curve
- [ ] **Analysis**: Analyze variance-bias tradeoff
- [ ] **Deliverable**: Push `model_selection.ipynb`

## DEEP LEARNING & DEPLOY WEEK (Days 15-21)
*Goal: Build minimal but impactful deep learning models and deploy them*

### Day 15: NEURAL NET FOUNDATIONS
- [ ] **Model**: Code a 3-layer perceptron from scratch using NumPy
- [ ] **View**: Plot training loss curve
- [ ] **Analysis**: Discuss vanishing gradients problem
- [ ] **Deliverable**: Push `neural_net_scratch.ipynb`

### Day 16: INTRO TO PYTORCH (or TF/Keras)
- [ ] **Model**: Re-implement Day 15 neural network with PyTorch
- [ ] **View**: Compare performance with NumPy implementation
- [ ] **Analysis**: Document benefits of using deep learning frameworks
- [ ] **Deliverable**: Push `pytorch_intro.ipynb`

### Day 17: CNN QUICKSTART
- [ ] **Model**: Build CNN for Fashion-MNIST classification (aim for 90%+ accuracy)
- [ ] **View**: Create grid of misclassified images
- [ ] **Analysis**: Identify patterns in misclassifications
- [ ] **Deliverable**: Push `cnn_fashion_mnist.ipynb`

### Day 18: TRANSFER LEARNING
- [ ] **Model**: Use pretrained ResNet to classify custom 20-image folder
- [ ] **View**: Generate Grad-CAM visualization for 2 samples
- [ ] **Analysis**: Discuss benefits of transfer learning vs training from scratch
- [ ] **Deliverable**: Push `transfer_learning.ipynb`

### Day 19: MLOPS LITE – MODEL PERSISTENCE
- [ ] **Model**: Save best model; load & infer in separate script
- [ ] **View**: Plot timing vs batch size
- [ ] **Analysis**: Document serialization options and tradeoffs
- [ ] **Deliverable**: Push `model_persistence.py` and saved model

### Day 20: FAST API DEPLOY
- [ ] **Model**: Wrap model with FastAPI
- [ ] **View**: Test locally via curl & Swagger UI
- [ ] **Analysis**: Measure latency vs batch size, document scalability notes
- [ ] **Deliverable**: Push `api.py` and deployment documentation

### Day 21: CAPSTONE & RETROSPECTIVE
- [ ] **Morning**: Select favorite dataset and build end-to-end solution
- [ ] **Afternoon**: Create UI/endpoint for model
- [ ] **Evening**: Write 1-page retrospective (achievements, gaps, next 30-day goals)
- [ ] **Deliverable**: Push `capstone` folder and `retrospective.md`

## Resources
- **Datasets**: Scikit-learn built-ins, Kaggle, UCI ML Repository
- **Documentation**: Python, NumPy, Pandas, Scikit-learn, PyTorch/TensorFlow
- **Books**: Python Data Science Handbook, Hands-On ML with Scikit-Learn & TensorFlow
- **Courses**: Fast.ai, Kaggle Learn, Coursera ML specializations

---

*Remember: The M-V-A framework ensures you build something concrete, visualize results, and critically analyze your work every day. Consistency is key!*
