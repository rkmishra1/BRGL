# Fast Validation Script for BRGL and Competing Methods

source("R/brgl.R")
source("R/competing_methods.R")

cat("\n--- Generating Toy Data ---\n")
set.seed(42)
n <- 30
p <- 8
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", 1:p)
beta_true <- c(3, -2, 1.5, 0, 0, 0, 0, 0)
y <- X %*% beta_true + rnorm(n, sd = 1.0)
y_ctr <- as.vector(y - mean(y))
X_std <- scale(X)

cat("\n--- Testing 1. Lasso ---\n")
b_lasso <- fit_lasso(X_std, y_ctr, k_folds = 3)
print(round(b_lasso, 4))

cat("\n--- Testing 2. Elastic Net ---\n")
b_en <- fit_en(X_std, y_ctr, k_folds = 3)
print(round(b_en, 4))

cat("\n--- Testing 3. OSCAR ---\n")
b_oscar <- fit_oscar(X_std, y_ctr, k_folds = 3)
print(round(b_oscar, 4))

cat("\n--- Testing 4. Bayesian Lasso (50 iterations) ---\n")
fit_bl <- BayesLasso(X_std, y_ctr, max_steps = 50, burn_in = 10, thin = 1, seed = 123)
print(round(fit_bl$beta, 4))

cat("\n--- Testing 5. Bayesian Elastic Net (50 iterations) ---\n")
fit_ben <- BayesElasticNet(X_std, y_ctr, max_steps = 50, burn_in = 10, thin = 1, seed = 123)
print(round(fit_ben$beta, 4))

cat("\n--- Testing 6. BRGL (50 iterations) ---\n")
fit_brgl <- brgl_mcmc(X_std, y_ctr, max_steps = 50, burn_in = 10, thin = 1, seed = 123)
print(round(fit_brgl$beta, 4))

cat("\n--- Validation Completed Successfully! ---\n")
