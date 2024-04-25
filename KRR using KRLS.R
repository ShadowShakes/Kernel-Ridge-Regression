# currently available kernel types are: 
# ['rbf', 'poly', 'linear']

library(KRLS)

# use simulated data generated from Python
data <- read.csv('/Users/orange/Documents/Practicum/Existing Code/data_from_Python.csv')
X <- as.matrix(data[, -which(names(data) == "y")])
y <- as.vector(data$y)


best_params <- read.csv('/Users/orange/Documents/Practicum/Existing Code/best_params_from_Python.csv')
best_gamma <- best_params$gamma


kernel_type <- "rbf"  


if (kernel_type == "rbf") {
  model <- krls(X, y, whichkernel = "gaussian", sigma = 1 / sqrt(2 * best_gamma))
} else if (kernel_type == "linear") {
  # has to manually set derivative as False?
  model <- krls(X, y, whichkernel = "linear", derivative = FALSE)
} else if (kernel_type == "poly") {
  # inconvenient to select degree
  best_degree <- best_params$degree
  model <- krls(X, y, whichkernel = "poly3", derivative = FALSE, sigma = 1 / sqrt(2 * best_gamma))
} else {
  stop(paste("Kernel", kernel_type, "is not supported."))
}


y_pred <- predict(model, X)$fit


plot(X[, 1], y, col = 'blue', pch = 16, main = paste('Kernel Ridge Regression with', kernel_type, 'Kernel'), xlab = 'X', ylab = 'Y', cex = 1)
points(X[, 1], y_pred, col = 'red', pch = 16, cex = 0.5)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), pch = 16)


mse <- mean((y - y_pred)^2)
rmse <- sqrt(mse)
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))