import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

# generate simulated non-linear data
rng = np.random.default_rng(321)
X = 10 * rng.random(2000)[:, np.newaxis]

y = (np.exp(-0.1 * X.ravel()) * np.sin(3 * X).ravel() +
     0.3 * X.ravel() * np.cos(X.ravel()) ** 2 +
     0.2 * np.log(X.ravel() + 2) ** 3)

y += 0.5 * rng.normal(size=X.shape[0])

# save Python data as .csv for further use in Rstudio
data = pd.DataFrame(np.hstack((X, y[:, np.newaxis])), columns=['X', 'y'])
data.to_csv('/Users/orange/Documents/Practicum/Existing Code/data_from_Python.csv', index=False)

kernel_type = "rbf"

# kernel ridge entity
krr = KernelRidge(kernel=kernel_type, gamma=0.1)

# grid search to find the optimal parameters
param_grid = {"alpha": [1e-2, 1e-1, 1, 10],
              "gamma": np.logspace(-2, 2, 5),
              "degree": [1, 2, 3]}
grid_search = GridSearchCV(krr, param_grid=param_grid, scoring="r2", cv=KFold(n_splits=5, shuffle=True, random_state=0))
grid_search.fit(X, y)

# best parameters
print(f"best parameters for Python krr: {grid_search.best_params_}")

# save the best parameters
best_params = grid_search.best_params_
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('/Users/orange/Documents/Practicum/Existing Code/best_params_from_Python.csv', index=False)

# prediction using krr
krr_best = grid_search.best_estimator_
X_plot = np.linspace(0, 10, 2000)[:, np.newaxis]
y_krr = krr_best.predict(X_plot)

# plot the results
plt.scatter(X, y, label='Data', color='black', s=5)
plt.plot(X_plot, y_krr, label='Kernel Ridge Regression ' + "using " + kernel_type, color='red')
plt.legend()
plt.show()

# calculate prediction accuracy
y_pred = krr_best.predict(X)

# MSE
mse = mean_squared_error(y, y_pred)
print("MSE:", mse)

# RMSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)
