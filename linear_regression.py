import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit
# Load the data
df = pd.read_csv('https://bit.ly/3cIH97A', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)
y = df.values[:, -1]
# Separate training and testing data
# This leaves a third of the data out for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
model = LinearRegression()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("R^2: %.3f" % result)


# Perform a simple linear regression with KFold validation
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model_k = LinearRegression()
results = cross_val_score(model_k, X, y, cv=kfold)
print(results)
print("MSE: mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))

# Perform a simple linear regression with ShuffleSplit validation
kfold = ShuffleSplit(n_splits=10, test_size=.33, random_state=7)
model_s = LinearRegression()
results = cross_val_score(model_s, X, y, cv=kfold)
print(results)
print("mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))