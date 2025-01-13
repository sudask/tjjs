import numpy as np
import pandas as pd

data = pd.read_csv("Credit.txt", sep=' ')
data.drop(columns="ID", inplace=True)

data['Gender'] = data['Gender'].apply(lambda gender: 0 if gender == " Male" else 1 if gender == "Female" else None)
data['Student'] = data['Student'].apply(lambda Student: 0 if Student == "Yes" else 1 if Student == "No" else None)
data['Married'] = data['Married'].apply(lambda Married: 0 if Married == "Yes" else 1 if Married == "No" else None)
data['Ethnicity'] = data['Ethnicity'].apply(lambda Ethnicity: 1 if Ethnicity == "Asian" else 2 if Ethnicity == "Caucasian" else 3)

# normalization
for column in data.columns:
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std

y = data['Balance']
data.pop('Balance')
X = data.copy()

# convert to ndarray
X = X.values
y = y.values

# define some functions
def softThres(x, λ):
    return np.sign(x) * np.maximum(np.abs(x) - λ, 0)

def lassoCoordinateDescent(X, y, λ, max_iter = 100000, tol = 1e-8):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)

    iteration = 0

    while iteration < max_iter:
        old_β = β.copy()

        for k in range(n_vars):
            c1 = (X[:, k] ** 2).sum()
            c2 = (X @ β - X[:, k] * β[k] - y) @ X[:, k]

            if -c2 < n_samples * -λ:
                β[k] = (-c2 + n_samples * λ) / c1
            elif -c2 > n_samples * λ:
                β[k] = (-c2 - n_samples * λ) / c1
            else:
                β[k] = 0
            

        iteration += 1
        if np.all(np.abs(β - old_β) < tol):
            break

    print("Results Coordinate Descent:")
    print("Iteration: ", iteration)
    print("Final Value: ", β)

def lassoProximalGradientDescent(X, y, λ, t=0.5, max_iter=100000, tol=1e-8):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)
    iteration = 0

    while iteration < max_iter:
        old_β = β.copy()

        gradient = (X.T @ (X @ β - y)) / n_samples
        β = softThres(β - t * gradient, t * λ)

        iteration += 1

        if np.all(np.abs(old_β - β) < tol):
            break

    print("Results Proximal Gradienct Descent:")
    print("Iteration: ", iteration)
    print("Final Value: ", β)

def lassoADMM(X, y, λ, rho=0.1, max_iter=100000, tol=1e-8):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)
    z = np.zeros(n_vars)
    u = np.zeros(n_vars)
    XTX = X.T @ X
    XTy = X.T @ y
    inv_matrix = np.linalg.inv(XTX / n_samples + rho * np.eye(n_vars))
    iteration = 0
    for i in range(max_iter):
        old_β = β.copy()

        new_β = inv_matrix @ (XTy / n_samples + rho * z - u)
        z = softThres(new_β + u / rho, λ / rho)
        u = u + rho * (new_β - z)
        β = new_β

        iteration += 1

        if np.all(np.abs(old_β - β) < tol):
            break
        
    print("Results ADMM:")
    print("Iteration: ", iteration)
    print("Final Value: ", β)


lassoCoordinateDescent(X, y, np.exp(-3))
lassoProximalGradientDescent(X, y, np.exp(-3))
lassoADMM(X, y, np.exp(-3))