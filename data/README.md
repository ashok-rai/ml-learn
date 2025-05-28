# Data Directory

This directory contains datasets used throughout the 21-day ML learning journey.

## Structure

- `sample/`: Small sample datasets for quick experimentation
  - `titanic_sample.csv`: Passenger information from the Titanic
  - `housing_sample.csv`: California housing price data
  - `iris_sample.csv`: Classic iris flower dataset

- `raw/`: Original, unprocessed datasets (to be added)
- `processed/`: Cleaned and transformed datasets (to be added)

## Sample Datasets

### Titanic Dataset
- **Description**: Passenger information from the Titanic disaster
- **Features**: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Target**: Survived (0 = No, 1 = Yes)
- **Use Case**: Binary classification

### Housing Dataset
- **Description**: California housing prices
- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: MedHouseVal (median house value)
- **Use Case**: Regression

### Iris Dataset
- **Description**: Measurements of iris flowers
- **Features**: sepal_length, sepal_width, petal_length, petal_width
- **Target**: species (setosa, versicolor, virginica)
- **Use Case**: Multi-class classification

## Adding New Datasets

When adding new datasets:

1. Place original files in the `raw/` directory
2. Document the source and any licensing information
3. Create processed versions in the `processed/` directory
4. Update this README with dataset descriptions

## Data Loading Examples

```python
# Load Titanic dataset
import pandas as pd

# Sample dataset
titanic_sample = pd.read_csv('data/sample/titanic_sample.csv')

# Full dataset (once added)
# titanic_full = pd.read_csv('data/raw/titanic.csv')
```
