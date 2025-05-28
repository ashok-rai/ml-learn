"""
Visualization Examples

This file contains examples of different visualization techniques using
Matplotlib, Seaborn, and Plotly. Use these as templates for your daily
visualization tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def save_fig(fig, filename, directory='visualizations'):
    """Save figure with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}_{timestamp}.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    return filepath

def basic_matplotlib_examples():
    """Basic Matplotlib visualization examples."""
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Example 1: Simple line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
    ax.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
    ax.set_title('Simple Line Plot', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "matplotlib_line_plot")
    
    # Example 2: Scatter plot with colormap
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x, y1, c=y2, cmap='viridis', s=50, alpha=0.8)
    ax.set_title('Scatter Plot with Colormap', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('sin(x)', fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('cos(x)', fontsize=12)
    save_fig(fig, "matplotlib_scatter_plot")
    
    # Example 3: Subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Line plot
    axs[0, 0].plot(x, y1, color='blue')
    axs[0, 0].set_title('Line Plot')
    
    # Subplot 2: Scatter plot
    axs[0, 1].scatter(x, y2, color='red', alpha=0.5)
    axs[0, 1].set_title('Scatter Plot')
    
    # Subplot 3: Bar plot
    x_bar = np.arange(5)
    y_bar = np.random.rand(5) * 10
    axs[1, 0].bar(x_bar, y_bar, color='green')
    axs[1, 0].set_title('Bar Plot')
    
    # Subplot 4: Histogram
    axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30, color='purple', alpha=0.7)
    axs[1, 1].set_title('Histogram')
    
    plt.tight_layout()
    save_fig(fig, "matplotlib_subplots")
    
    return "Matplotlib examples created"

def seaborn_examples():
    """Seaborn visualization examples."""
    # Generate sample data
    np.random.seed(42)
    tips = sns.load_dataset('tips')
    iris = sns.load_dataset('iris')
    
    # Example 1: Distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(tips['total_bill'], kde=True, ax=ax)
    ax.set_title('Distribution of Total Bill', fontsize=16)
    save_fig(fig, "seaborn_histplot")
    
    # Example 2: Scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='total_bill', y='tip', data=tips, ax=ax)
    ax.set_title('Relationship between Bill and Tip', fontsize=16)
    save_fig(fig, "seaborn_regplot")
    
    # Example 3: Categorical plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='day', y='total_bill', hue='sex', data=tips, ax=ax)
    ax.set_title('Total Bill by Day and Gender', fontsize=16)
    save_fig(fig, "seaborn_boxplot")
    
    # Example 4: Pair plot
    pair_plot = sns.pairplot(iris, hue='species', height=2.5)
    save_fig(pair_plot, "seaborn_pairplot")
    
    # Example 5: Heatmap
    corr = iris.drop('species', axis=1).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=16)
    save_fig(fig, "seaborn_heatmap")
    
    return "Seaborn examples created"

def plotly_examples():
    """
    Plotly examples - note these will be displayed in a notebook
    but saved as HTML files here.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import os
        
        # Create directory for plotly outputs
        os.makedirs('visualizations/plotly', exist_ok=True)
        
        # Example 1: Line plot
        df = px.data.stocks()
        fig = px.line(df, x='date', y=df.columns[1:], title='Stock Prices')
        fig.write_html('visualizations/plotly/plotly_line_plot.html')
        
        # Example 2: Scatter plot
        df = px.data.iris()
        fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                         size='petal_length', hover_data=['petal_width'])
        fig.write_html('visualizations/plotly/plotly_scatter_plot.html')
        
        # Example 3: Bar chart
        df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
        fig = px.bar(df, x='country', y='pop', color='lifeExp',
                     title='Population of European Countries')
        fig.write_html('visualizations/plotly/plotly_bar_chart.html')
        
        # Example 4: 3D Scatter plot
        df = px.data.iris()
        fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                            color='species')
        fig.write_html('visualizations/plotly/plotly_3d_scatter.html')
        
        # Example 5: Interactive dashboard
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Line Plot', 'Bar Chart', 'Scatter Plot', 'Pie Chart'))
        
        # Add traces
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
        fig.add_trace(go.Bar(x=[1, 2, 3], y=[7, 8, 9]), row=1, col=2)
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 11, 12], mode='markers'), row=2, col=1)
        fig.add_trace(go.Pie(labels=['A', 'B', 'C'], values=[1, 2, 3]), row=2, col=2)
        
        fig.update_layout(height=800, width=800, title_text="Interactive Dashboard")
        fig.write_html('visualizations/plotly/plotly_dashboard.html')
        
        return "Plotly examples created"
    except ImportError:
        return "Plotly not installed. Install with: pip install plotly"

if __name__ == "__main__":
    print("Creating visualization examples...")
    basic_matplotlib_examples()
    seaborn_examples()
    plotly_result = plotly_examples()
    print(plotly_result)
    print("Done!")
