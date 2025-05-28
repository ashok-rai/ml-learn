"""
Sample Model Template

This file provides a basic structure for creating and saving ML models.
It can be used as a starting point for your own model implementations.
"""

import numpy as np
import pickle
import os
from datetime import datetime


class SimpleLinearRegression:
    """A simple linear regression model implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.training_history = {
            'loss': []
        }
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : returns an instance of self.
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Make predictions
            y_pred = self._predict(X)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate loss
            loss = self._mse(y, y_pred)
            self.training_history['loss'].append(loss)
            
            # Print progress every 100 iterations
            if (i+1) % 100 == 0:
                print(f'Iteration {i+1}/{self.n_iterations} | Loss: {loss:.4f}')
        
        return self
    
    def _predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Make predictions for new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        return self._predict(X)
    
    def _mse(self, y_true, y_pred):
        """Calculate mean squared error."""
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X, y):
        """Calculate R^2 score."""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def save(self, filepath=None):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the model. If None, a default path is used.
        
        Returns:
        --------
        filepath : str
            Path where the model was saved
        """
        if filepath is None:
            # Create a default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(os.getcwd(), f'simple_linear_model_{timestamp}.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        
        Returns:
        --------
        model : SimpleLinearRegression
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.5
    
    # Create and train the model
    model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate score
    score = model.score(X, y)
    print(f"R^2 Score: {score:.4f}")
    
    # Save the model
    model.save("models/simple_linear_model.pkl")
