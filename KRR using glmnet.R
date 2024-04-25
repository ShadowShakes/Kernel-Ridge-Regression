# currently available kernel types are: 
# ['rbf', 'poly', 'polynomial', 'linear', 'cosine', 'sigmoid', 'laplacian', 'additive_chi2', 'chi2']

library(glmnet)

kernel_type <- "rbf"  # change this to use different kernels

# functionality to calculate various kernel matrices
calculate_kernel_matrix <- function(X, kernel_type, gamma, degree, coef0 = 1) {
  if (kernel_type == "linear") {
    return(X %*% t(X))
  } else if (kernel_type == "rbf") {
    sq_dist <- as.matrix(dist(X))^2
    return(exp(-gamma * sq_dist))
  } else if (kernel_type == "poly") {
    return((gamma * (X %*% t(X)) + coef0)^degree)
  } else if (kernel_type == "cosine") {
    X_normalized <- X / sqrt(rowSums(X^2))
    return(X_normalized %*% t(X_normalized))
  } else if (kernel_type == "chi2") {
    result <- matrix(0, nrow(X), nrow(X))
    for (i in 1:nrow(X)) {
      for (j in 1:nrow(X)) {
        summand <- (X[i, ] - X[j, ])^2 / (X[i, ] + X[j, ])
        summand[is.nan(summand)] <- 0
        result[i, j] <- sum(summand)
      }
    }
    return(exp(-gamma * result))
  } else if (kernel_type == "additive_chi2") {
    result <- matrix(0, nrow(X), nrow(X))
    for (i in 1:nrow(X)) {
      for (j in 1:nrow(X)) {
        summand <- X[i, ] * X[j, ] / (X[i, ] + X[j, ])
        summand[is.nan(summand)] <- 0
        result[i, j] <- sum(summand)
      }
    }
    return(result)
  } else if (kernel_type == "sigmoid") {
    return(tanh(gamma * (X %*% t(X)) + coef0))
  } else if (kernel_type == "laplacian") {
    result <- matrix(0, nrow(X), nrow(X))
    for (i in 1:nrow(X)) {
      for (j in 1:nrow(X)) {
        result[i, j] <- sum(abs(X[i, ] - X[j, ]))
      }
    }
    return(exp(-gamma * result))
  } else {
    stop(paste("Kernel", kernel_type, "is not supported."))
  }
}


# use simulated data generated from Python
data <- read.csv('/Users/orange/Documents/Practicum/Existing Code/data_from_Python.csv')
X <- as.matrix(data$X)
y <- as.vector(data$y)


# read the best parameters from Python to test if the results are the same under the same parameters
best_params <- read.csv('/Users/orange/Documents/Practicum/Existing Code/best_params_from_Python.csv')
best_alpha <- best_params$alpha
best_gamma <- best_params$gamma
best_degree <- best_params$degree


K <- calculate_kernel_matrix(X, kernel_type, gamma = best_gamma, degree = best_degree)


# train the model using the computed kernel matrix
model <- glmnet(K, y, alpha = 0, lambda = best_alpha)  # lambda should be tuned


# predict using the model (note that you need to compute the kernel matrix for new data points)
y_pred <- predict(model, newx = K)


# plot the prediction
plot(X[,1], y, col = 'blue', pch = 16, main = paste('Kernel Ridge Regression with', kernel_type, 'Kernel'), xlab = 'X', ylab = 'Y', cex = 1)
points(X[,1], y_pred, col = 'red', pch = 16, cex = 0.5)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), pch = 16)


# calculate MSE and RMSE
mse <- mean((y - y_pred)^2)
rmse <- sqrt(mse)
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))

