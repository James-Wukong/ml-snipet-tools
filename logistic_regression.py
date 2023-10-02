import sympy as sp
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression

b0, b1, x = sp.symbols('b0 b1 x')
p = 1.0 / (1.0 + sp.exp(-(b0 + b1 * x)))
p = p.subs(b0, -2.823)
p = p.subs(b1, 0.620)

print(p)
sp.plot(p)

# Load the data
df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)
y = df.values[:, -1]
# Perform logistic regression
# Turn off penalty
model = LogisticRegression(penalty=None)
model.fit(X, y)
# print beta1
print(model.coef_.flatten()) # 0.69267212
# print beta0
print(model.intercept_.flatten()) # -3.17576395


patient_data = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",").itertuples()
b0 = -3.17576395
b1 = 0.69267212
def logistic_function(x):
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p
# Calculate the joint likelihood
joint_likelihood = 1.0
for p in patient_data:
    print(f'joint likelihood is {joint_likelihood}, lr is {logistic_function(p.x)}\n')
    if p.y == 1.0:
        joint_likelihood *= logistic_function(p.x)
    elif p.y == 0.0:
        joint_likelihood *= (1.0 - logistic_function(p.x))
print(joint_likelihood) 

# 4.7911180221699105e-05