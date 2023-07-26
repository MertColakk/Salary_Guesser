import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Datasets/Salary.csv")

dataset_x = dataset.iloc[:,:-1].values
dataset_y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(dataset_x,dataset_y,random_state=0,test_size=0.2)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary and Experience Guess(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary and Experience Guess(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
