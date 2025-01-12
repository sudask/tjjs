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

def lassoCoordinateDescent(X, y, λ, max_iter = 1000, tol = 1e-20):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)

    iteration = 0

    while iteration < max_iter:
        new_β = np.zeros(n_vars)

        for k in range(n_vars):
            c1 = (X[:, k] ** 2).sum()
            c2 = (X @ β - X[:, k] * β[k] - y) @ X[:, k]

            if -c2 < n_samples * -λ:
                β[k] = (-c2 + n_samples * λ) / c1
            elif -c2 > n_samples * λ:
                β[k] = (-c2 - n_samples * λ) / c1
            else:
                β[k] = 0
            

        if np.all(np.abs(new_β - β) < tol):
            break

        iteration += 1

    return β

def lassoProximalGradientDescent(X, y, λ, t=0.05, max_iter=1000, tol=1e-20):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)
    iteration = 0

    while iteration < max_iter:
        gradient = (X.T @ (X @ β - y)) / n_samples
        new_β = softThres(β - t * gradient, t * λ)

        if np.all(np.abs(new_β - β) < tol):
            break

        β = new_β
        iteration += 1

    return β

def lassoADMM(X, y, λ, rho=1.0, max_iter=1000, tol=1e-4):
    n_samples, n_vars = X.shape
    β = np.zeros(n_vars)
    z = np.zeros(n_vars)
    u = np.zeros(n_vars)
    XTX = X.T @ X
    XTy = X.T @ y
    inv_matrix = np.linalg.inv(XTX / n_samples + rho * np.eye(n_vars))
    for i in range(max_iter):
        β = inv_matrix @ (XTy / n_samples + rho * z - u)
        z_new = softThres(β + u / rho, λ / rho)
        u = u + rho * (β - z_new)

        if np.linalg.norm(z_new - z) < tol:
            break
        z = z_new
    return β


res1 = lassoCoordinateDescent(X, y, np.exp(-3))
res2 = lassoProximalGradientDescent(X, y, np.exp(-3))
res3 = lassoADMM(X, y, np.exp(-3))
print(res1)
print(res2)
print(res3)