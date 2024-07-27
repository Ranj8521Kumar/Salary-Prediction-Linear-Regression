# Simple Linear Regression Model

This project demonstrates how to build a simple linear regression model to predict salaries based on years of experience using Python. The steps include:

1. Importing the necessary libraries
2. Loading and preparing the dataset
3. Splitting the dataset into training and test sets
4. Training the linear regression model
5. Making predictions on the test set
6. Visualizing the results
7. Predicting the salary for a given number of years of experience

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/Ranj8521Kumar/Salary-Prediction-Linear-Regression.git
    cd SimpleLinearRegression
    ```
2. Ensure you have the required libraries installed:
    ```sh
    pip install numpy matplotlib pandas scikit-learn
    ```
3. Run the Python script:
    ```sh
    python model.py
    ```
## Step-by-Step Guide

### Step 1: Importing Libraries
First, we import the essential libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Step 2: Importing the Dataset
Next, we load our dataset and separate the independent and dependent variables:
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### Step 3: Splitting the Dataset
We split our dataset into training and test sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Step 4: Training the Model
We train our linear regression model using the training set:
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
### Step 5: Making Predictions
Now, we use the trained model to make predictions on the test set:
```python
y_pred = regressor.predict(X_test)
```

### Step 6: Visualizing the Training Set Results
We visualize the training set results to see the regression line:
```python
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```
![image](https://github.com/user-attachments/assets/fe4069d2-234d-466b-9609-8fb89b80fa51)

### Step 7: Visualizing the Test Set Results
We visualize the Test set results to see the regression line:
```python
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

![image](https://github.com/user-attachments/assets/13a8dd60-415b-4e72-bf37-0b40b3148920)



### Step 8: User Input Prediction
Finally, let's make it interactive by taking user input:
```python
years_of_experience = float(input("Enter years of experience: "))
predicted_salary = regressor.predict([[years_of_experience]])
print("predicted Salary for the years of ",years_of_experience, "is", predicted_salary)
```
###OutPut: predicted Salary for the years of  15.0 is [159385.23868361]

## Example
You can predict the salary for a given number of years of experience by running the script and entering the number of years when prompted.

Happy coding! 😊
