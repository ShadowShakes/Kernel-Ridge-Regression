# currently available kernel types are: 
# ['rbf', 'poly', 'polynomial', 'linear', 'cosine', 'sigmoid', 'laplacian', 'additive_chi2', 'chi2']

kernel_type = "rbf"  # change this to use different kernels

# helper function to check if a value is in a specific interval
is_in_interval <- function(x, lower = -Inf, upper = Inf, lower_inclusive = TRUE, upper_inclusive = TRUE) {
  if (lower_inclusive) {
    lower_ok <- x >= lower
  } else {
    lower_ok <- x > lower
  }
  if (upper_inclusive) {
    upper_ok <- x <= upper
  } else {
    upper_ok <- x < upper
  }
  lower_ok & upper_ok
}


# helper function for cosine kernel calculation
cosine_similarity <- function(X, Y = NULL) {
  if (is.null(Y)) Y <- X
  X_normalized <- X / sqrt(rowSums(X^2))
  Y_normalized <- Y / sqrt(rowSums(Y^2))
  similarity_matrix <- X_normalized %*% t(Y_normalized)
  return(similarity_matrix)
}


# define the kernels to be used
calculate_kernel <- function(X, Y = NULL, kernel , gamma, degree, coef0 = 0, ...) {
  if (is.null(Y)) Y <- X
  
  # check parameter constraints
  if (!is.null(gamma) && !is_in_interval(gamma, 0, Inf, lower_inclusive = FALSE)) {
    stop("gamma must be greater than 0 or NULL")
  }
  if (!is_in_interval(degree, 0, Inf, lower_inclusive = FALSE)) {
    stop("degree must be an integer greater than 0")
  }
  if (!is.null(coef0) && !is_in_interval(coef0, -Inf, Inf, lower_inclusive = FALSE, upper_inclusive = FALSE)) {
    stop("coef0 can be any real number")
  }
  
  # calculate kernel results
  if (kernel == "linear") {
    as.matrix(X) %*% t(Y)
  } else if (kernel == "polynomial" || kernel == "poly") {
    (gamma * as.matrix(X) %*% t(Y) + coef0) ^ degree
  } else if (kernel == "rbf") {
    sq_dist <- as.matrix(dist(rbind(X, Y)))^2
    exp(-gamma * sq_dist[1:nrow(X), (nrow(X) + 1):(nrow(X) + nrow(Y))])
  } else if (kernel == "laplacian") {
    laplace_dist <- as.matrix(dist(rbind(X, Y), method = "manhattan"))
    exp(-gamma * laplace_dist[1:nrow(X), (nrow(X) + 1):(nrow(X) + nrow(Y))])
  } else if (kernel == "sigmoid") {
    tanh(gamma * (as.matrix(X) %*% t(Y)) + coef0)
  } else if (kernel == "cosine") {
    cosine_similarity(X, Y)
  } else if (kernel == "chi2") {
    result <- matrix(0, nrow(X), nrow(Y))
    for (i in 1:nrow(X)) {
      for (j in 1:nrow(Y)) {
        x <- X[i, ]
        y <- Y[j, ]
        summand <- (x - y)^2 / (x + y)
        summand[is.nan(summand)] <- 0  # in case that x + y = 0
        result[i, j] <- sum(summand)
      }
    }
    exp(-gamma * result)
  } else if (kernel == "additive_chi2") {
    # here the additive chi-squared kernel is computed as k(x, y) = -sum((x - y)^2 / (x + y))
    matrix(apply(expand.grid(1:nrow(X), 1:nrow(Y)), 1, function(index) {
      x <- X[index[1], ]
      y <- Y[index[2], ]
      - sum((x - y)^2 / (x + y))
    }), nrow = nrow(X))
  } else {
    stop("sorry but the input kernel type is currently not supported!")
  }
}


# kernel ridge regression functionality
kernel_ridge <- function(X, y, alpha, kernel, gamma, degree, coef0=0) {
  # check parameter constraints
  if (!is_in_interval(alpha, 0, Inf, lower_inclusive = FALSE)) {
    stop("alpha must be greater than 0")
  }
  
  if (kernel == "sigmoid" || kernel == "poly" || kernel == "polynomial") {
    coef0 = 1
  }
  K <- calculate_kernel(X, kernel = kernel, gamma = gamma, degree = degree, coef0 = coef0)
  n <- nrow(K)
  I <- diag(n)
  beta <- solve(K + alpha * I) %*% y
  return(list(coef = beta, X = X, kernel = kernel, gamma = gamma, degree = degree, coef0 = coef0))
}


# prediction functionality
predict_kernel_ridge <- function(model, X) {
  K <- calculate_kernel(X, model$X, model$kernel, model$gamma, model$degree, model$coef0)
  return(K %*% model$coef)
}




# use simulated data generated from Python
data <- read.csv('/Users/orange/Documents/Practicum/Existing Code/data_from_Python.csv')
X <- as.matrix(data$X)
y <- data$y
y <- matrix(y, ncol = 1)


# # cross-validation to select best parameters (optional)
# calculate_r_squared <- function(y_true, y_pred, p) {
#   sst <- sum((y_true - mean(y_true))^2)
#   ssr <- sum((y_true - y_pred)^2)
#   r_squared <- 1 - (ssr / sst)
#   return(r_squared)
# }
# 
# 
# # parameters grids
# alpha_grid <- c(1e-2, 1e-1, 1, 10)
# gamma_grid <- 10^seq(-2, 2, length.out = 5)
# degree_grid <- 1:3
# 
# perform_grid_search_r2 <- function(X, y, alpha_grid, gamma_grid, degree_grid, cv_folds = 5) {
#   best_score <- -Inf
#   best_params <- list(alpha = NA, gamma = NA, degree = NA)
# 
#   for (alpha in alpha_grid) {
#     for (gamma in gamma_grid) {
#       for (degree in degree_grid) {
#         r2_scores <- numeric(cv_folds)
#         set.seed(0)
#         for (fold in 1:cv_folds) {
#           indices <- sample(seq_along(y))
#           X_train <- X[indices[1:(length(indices) * 0.8)], , drop = FALSE]
#           y_train <- y[indices[1:(length(indices) * 0.8)]]
#           X_test <- X[indices[(length(indices) * 0.8 + 1):length(indices)], , drop = FALSE]
#           y_test <- y[indices[(length(indices) * 0.8 + 1):length(indices)]]
# 
#           model <- kernel_ridge(X_train, y_train, alpha = alpha, kernel = kernel_type, gamma = gamma, degree = degree)
# 
#           y_pred <- predict_kernel_ridge(model, X_test)
#           r2_scores[fold] <- calculate_r_squared(y_test, y_pred)
#         }
# 
#         mean_r2_score <- mean(r2_scores)
# 
#         if (mean_r2_score > best_score) {
#           best_score <- mean_r2_score
#           best_params <- list(alpha = alpha, gamma = gamma, degree = degree)
#         }
#       }
#     }
#   }
# 
#   return(list(best_params = best_params, best_score = best_score))
# }
# 
# 
# grid_search_results_r2 <- perform_grid_search_r2(X, y, alpha_grid, gamma_grid, degree_grid)
# print(grid_search_results_r2$best_params)
# print(paste("Best R^2 Score:", grid_search_results_r2$best_score))
# 
# 
# best_alpha = grid_search_results_r2$best_params$alpha
# best_gamma = grid_search_results_r2$best_params$gamma
# best_degree = grid_search_results_r2$best_params$degree




# read the best parameters from Python to test if the results are the same under the same parameters
best_params <- read.csv('/Users/orange/Documents/Practicum/Existing Code/best_params_from_Python.csv')
best_alpha <- best_params$alpha
best_gamma <- best_params$gamma
best_degree <- best_params$degree


# fit model using our kernel_ridge functionality using the best parameters from Python
model <- kernel_ridge(X, y, alpha = best_alpha, kernel = kernel_type, gamma = best_gamma, degree = best_degree)


# prediction
X_pred <- matrix(seq(0, 10, length.out = 2000), ncol = 1)
y_pred <- predict_kernel_ridge(model, X_pred)


# plot the prediction results
plot(X, y, col = 'blue', main = paste('Kernel Ridge Regression using', kernel_type), xlab = 'X', ylab = 'Y',pch = 16)
lines(X_pred, y_pred, col = 'red', lwd = 3)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 2)


# predict using the model
y_train_pred <- predict_kernel_ridge(model, X)


# MSE
mse <- mean((y - y_train_pred)^2)
print(paste("MSE:", mse))


# RMSE
rmse <- sqrt(mse)
print(paste("RMSE:", rmse))
