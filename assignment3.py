import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import statistics 
#Part A
df = pd.read_csv("assignment3_data.csv" , comment='#')
X1=df.iloc[:,0]
X2=df.iloc[:,1]
Y = df.iloc[:,2]
X = np.column_stack((X1,X2))

#Part B+C
poly = sklearn.preprocessing.PolynomialFeatures(degree=5)
X_new_features = poly.fit_transform(X)

x_range = []
grid = np.linspace(-2.5,2.5)
for i in grid:
    for j in grid:
        x_range.append([i,j])
x_range = np.array(x_range)
x_range_poly = poly.fit_transform(x_range)

C_values_Lasso=[1, 10, 100, 1000]
predictions_lasso =[]
print("*************************** Lasso predictions ***************************************")
for i in C_values_Lasso:
    new_model = linear_model.Lasso(alpha=(1/(2*i)))
    new_model.fit(X_new_features, Y)
    predictions_lasso.append(new_model.predict(x_range_poly))
    print("model C = ", i)
    print("intercept ", new_model.intercept_)
    print("Coef ", new_model.coef_)

x_range_transpose = np.transpose(x_range)

#Part E
C_values_Ridge=[0.5, 1, 10, 100, 1000]
predictions_ridge =[]
print("*************************** ridge predictions ***************************************")
for i in C_values_Ridge:
    new_model = linear_model.Ridge(alpha=(1/(2*i)))
    new_model.fit(X_new_features, Y)
    predictions_ridge.append(new_model.predict(x_range_poly))
    print("model C = ", i)
    print("intercept ", new_model.intercept_)
    print("Coef ", new_model.coef_)

#Part ii
#Part A
C = 1

splits = [2, 5, 10, 25, 50, 100]
mean_values1_Lasso = []
standard_deviation_values1_Lasso =[]
for i in splits:
    new_estimates = []
    kf = KFold(n_splits = i)
    for train1, test1, in kf.split(X):
        new_model = linear_model.Lasso(alpha=(1/(2*C)))
        new_model.fit(X[train1], Y[train1])
        prediction = new_model.predict(X[test1])
        new_estimates.append(mean_squared_error(prediction, Y[test1]))

    mean = sum(new_estimates)/i
    standard_deviation = statistics.stdev(new_estimates)
    mean_values1_Lasso.append(mean)
    standard_deviation_values1_Lasso.append(standard_deviation)

print("*************************** Lasso values ***************************************")
print("mean values", mean_values1_Lasso)
print("standard deviation values", standard_deviation_values1_Lasso)

#part B
mean_values2_Lasso = []
standard_deviation_values2_Lasso =[]
for i in C_values_Lasso:
    new_estimates = []
    kf = KFold(n_splits = 5)
    for train1, test1, in kf.split(X):
        new_model = linear_model.Lasso(alpha=(1/(2*C)))
        new_model.fit(X[train1], Y[train1])
        prediction = new_model.predict(X[test1])
        new_estimates.append(mean_squared_error(prediction, Y[test1]))

    mean = sum(new_estimates)/i
    standard_deviation = statistics.stdev(new_estimates)
    mean_values2_Lasso.append(mean)
    standard_deviation_values2_Lasso.append(standard_deviation)

print("mean values", mean_values2_Lasso)
print("standard deviation values", standard_deviation_values2_Lasso)

#part D
mean_values2_Ridge = []
standard_deviation_values2_Ridge =[]
for i in C_values_Ridge:
    new_estimates = []
    kf = KFold(n_splits = 5)
    for train1, test1, in kf.split(X):
        new_model = linear_model.Ridge(alpha=(1/(2*C)))
        new_model.fit(X[train1], Y[train1])
        prediction = new_model.predict(X[test1])
        new_estimates.append(mean_squared_error(prediction, Y[test1]))

    mean = sum(new_estimates)/i
    standard_deviation = statistics.stdev(new_estimates)
    mean_values2_Ridge.append(mean)
    standard_deviation_values2_Ridge.append(standard_deviation)
print("*************************** Ridge values ***************************************")
print("mean values", mean_values2_Ridge)
print("standard deviation values", standard_deviation_values2_Ridge)

#graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X1")
ax.set_ylabel('X2')
ax.set_zlabel("Y")
plt.title("raw data")
ax.scatter(X1, X2, Y)

for a in range(0,4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X1")
    ax.set_ylabel('X2')
    ax.set_zlabel("Y")
    plt.title("C={} Predictions - Lasso".format(C_values_Lasso[a]))
    ax.plot_trisurf(x_range_transpose[0], x_range_transpose[1], predictions_lasso[a], alpha=0.5)
    ax.scatter(X1, X2, Y, color='r')
plt.show()

for a in range(0,5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X1")
    ax.set_ylabel('X2')
    ax.set_zlabel("Y")
    plt.title("C={} Predictions - Ridge".format(C_values_Ridge[a]))
    ax.plot_trisurf(x_range_transpose[0], x_range_transpose[1], predictions_ridge[a], alpha=0.5)
    ax.scatter(X1, X2, Y, color='r')
plt.show()

fig = plt.figure()
plt.title("Lasso - Folds vs Average Mean Squared Errors With Standard Deviation")
plt.xlabel("Number of Splits")
plt.ylabel("Average of Mean Squared Errors")
plt.errorbar(splits, mean_values1_Lasso, yerr=standard_deviation_values1_Lasso)

fig = plt.figure()
plt.title("Lasso - C Value vs Average Mean Squared Errors With Standard Deviation")
plt.errorbar(C_values_Lasso, mean_values2_Lasso, yerr=standard_deviation_values2_Lasso)
plt.xscale("log")
plt.xlabel("C Values")
plt.ylabel("Average of Mean Squared Errors")
plt.show()

fig = plt.figure()
plt.title("Ridge - C Value vs Average Mean Squared Errors With Standard Deviation")
plt.errorbar(C_values_Ridge, mean_values2_Ridge, yerr=standard_deviation_values2_Ridge)
plt.xscale("log")
plt.xlabel("C Values")
plt.ylabel("Average of Mean Squared Errors")
plt.show()