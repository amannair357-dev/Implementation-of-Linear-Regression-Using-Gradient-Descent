# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset and separate the input features and target variable.
2. Standardize the input features and target using StandardScaler.
3. Initialize the model parameters (theta) to zero and add a bias column.
4. Apply Gradient Descent repeatedly to update theta by minimizing prediction error.
5. Scale the new input, predict the profit, and convert the prediction back to the original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Aman Nair
RegisterNumber:  212225240008


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Linear regression using Gradient Descent
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  #add bias column
  X=np.c_[np.ones(len(X1)),X1]
  #initialize theta
  theta=np.zeros((X.shape[1],1))

  #Gradient Descent
  for _ in range(num_iters):
    predictions=X.dot(theta)
    errors=predictions-y
    theta -= learning_rate*((1/len(X1))*X.T.dot(errors))
  return theta

#Load dataset
data=pd.read_csv(r"/content/50_Startups.csv")

#Features (R&D spend,administration,marketing spend)
X=data.iloc[::-1].values

#Remove categorical column(State) if present
X=X[:,[0,1,2]]

#Target
y=data.iloc[:,-1].values.reshape(-1,1)

#Scale Features
scaler_X=StandardScaler()
X_scaled=scaler_X.fit_transform(X)

#Scale Target
scaler_y=StandardScaler()
y_scaled= scaler_y.fit_transform(y)
print(X_scaled)

#Train model
theta=linear_regression(X_scaled,y_scaled)

#New data
new_data=np.array([[165349.2,136897.8,471784.1]])
new_scaled=scaler_X.transform(new_data)
new_scaled=np.c_[np.ones(1),new_scaled]
prediction_scaled=new_scaled.dot(theta)
prediction=scaler_y.inverse_transform(prediction_scaled)
print("Scaled prediction:",prediction_scaled)
print("Predicted profit:",prediction)

*/
```

## Output:

<img width="630" height="788" alt="Screenshot 2026-08-18 093918" src="https://github.com/user-attachments/assets/1ca53039-31eb-4d67-b401-10fac8515806" />
<img width="656" height="371" alt="Screenshot 2026-08-18 093924" src="https://github.com/user-attachments/assets/c7ab92ed-c95d-4046-89cb-0d83c08b4efa" />
<img width="295" height="187" alt="Screenshot 2026-08-18 093935" src="https://github.com/user-attachments/assets/8a1c93c7-d589-485b-9bd6-4fec7111d7ca" />
<img width="470" height="80" alt="Screenshot 2026-08-18 093940" src="https://github.com/user-attachments/assets/a1508b8e-975f-400b-ab09-8d0e508fa844" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
