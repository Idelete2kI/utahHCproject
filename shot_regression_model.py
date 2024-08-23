
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/neelc/Desktop/ARIshots_cleaned.csv")
df_cleaned = df.loc[~(df == 0).any(axis=1)]
df = df_cleaned

df1 = pd.read_csv("C:/Users/neelc/Desktop/shots_ropes_with_xG.csv")

x1 =  np.array(df['5v5 xg (g%)']).reshape(-1,1)
x2 = np.array(df['5v5 xg (avg)']).reshape(-1,1)
y = np.array(df['5v5 xGoal'])

x = np.hstack((x1,x2))

model = LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(mse)
print(r2)

# Get the model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print the regression equation
print(f"Regression Equation: Y = {intercept:.4f} + ({coefficients[0]:.4f})*X1 + ({coefficients[1]:.4f})*X2")

#Give each of my shots new xG value based on regression equation
values = [coefficients[0] * df1['5v5 xG (g%)'].iloc[i] for i in range(len(df1))]
arr1 = np.array(values)
sum1 = np.sum(arr1)

values2 = [coefficients[1] * df1['5v5 xG (avg)'].iloc[i] for i in range(len(df1))]
arr2 = np.array(values2)
sum2 = np.sum(arr2)

print(sum2+sum1+intercept)
  





plt.scatter(y, y_pred, color='blue')
plt.plot(y, y, color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.xlabel('Better Model Expected Goals')
plt.ylabel('Predicted Expected Goals')
plt.title('Comparison of Model Predictions')
plt.legend()
plt.show()