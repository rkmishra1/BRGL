# Competing Methods for BRGL Comparison
# Implements:
# 1. Lasso and Elastic Net (via glmnet)
# 2. Optimized OSCAR (via FISTA and PAVA)
# 3. Bayesian Lasso (Park & Casella, 2008)
# 4. Bayesian Elastic Net (Kyung et al., 2010)

library(glmnet)

# -------------------------------------------------------------
# 1. Lasso & Elastic Net Wrapper Functions
# -------------------------------------------------------------

fit_lasso <- function(X, y, k_folds = 5) {
  cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = k_folds)
  beta <- as.vector(predict(cv_fit, type = "coefficients", s = "lambda.min"))[-1]
  names(beta) <- colnames(X)
  return(beta)
}

fit_en <- function(X, y, k_folds = 5) {
  alphas <- seq(0.1, 0.9, by = 0.2)
  best_err <- Inf
  best_alpha <- 0.5
  best_lambda <- 0.1
  
  for (a in alphas) {
    fit <- cv.glmnet(X, y, alpha = a, nfolds = k_folds)
    min_err <- min(fit$cvm)
    if (min_err < best_err) {
      best_err <- min_err
      best_alpha <- a
      best_lambda <- fit$lambda.min
    }
  }
  
  final_fit <- glmnet(X, y, alpha = best_alpha, lambda = best_lambda)
  beta <- as.vector(coef(final_fit))[-1]
  names(beta) <- colnames(X)
  return(beta)
}

# -------------------------------------------------------------
# 2. OSCAR Implementation (FISTA + PAVA, Highly Optimized)
# -------------------------------------------------------------

# Pool Adjacent Violators Algorithm (PAVA) for descending order
pava_descending <- function(z) {
  p <- length(z)
  values <- z
  weights <- rep(1, p)
  active <- rep(TRUE, p)
  
  i <- 1
  while (i < p) {
    if (active[i]) {
      next_active <- which(active[(i+1):p])[1] + i
      if (is.na(next_active)) break
      
      if (values[i] < values[next_active]) {
        w_sum <- weights[i] + weights[next_active]
        val_new <- (values[i] * weights[i] + values[next_active] * weights[next_active]) / w_sum
        values[i] <- val_new
        weights[i] <- w_sum
        active[next_active] <- FALSE
        if (i > 1) {
          prev_active <- tail(which(active[1:(i-1)]), 1)
          i <- if (length(prev_active) > 0) prev_active else 1
        } else {
          i <- 1
        }
      } else {
        i <- next_active
      }
    } else {
      i <- i + 1
    }
  }
  
  out <- numeric(p)
  curr_val <- NA
  for (i in 1:p) {
    if (active[i]) curr_val <- values[i]
    out[i] <- curr_val
  }
  return(pmax(out, 0))
}

# Proximal operator of the OSCAR penalty
prox_oscar <- function(v, lambda1, lambda2) {
  p <- length(v)
  abs_v <- abs(v)
  ord <- order(abs_v, decreasing = TRUE)
  u <- abs_v[ord]
  
  w <- lambda1 + lambda2 * (p - 1:p)
  z <- pmax(u - w, 0)
  
  x_proj <- pava_descending(z)
  
  out <- numeric(p)
  out[ord] <- sign(v[ord]) * x_proj
  return(out)
}

# Optimized FISTA solver for OSCAR
# Accepts precomputed XtX, Xty, Lipschitz constant L, and sample size n
fit_oscar_fixed <- function(XtX, Xty, lambda1, lambda2, L, n, max_iter = 1000, tol = 1e-6) {
  p <- ncol(XtX)
  t_step <- 1 / L
  
  beta <- numeric(p)
  beta_prev <- beta
  y_acc <- beta
  t_acc <- 1
  
  for (k in 1:max_iter) {
    # Gradient of 1/(2n) * ||y - X*beta||_2^2 is - (X'y - X'X*beta) / n
    grad <- -(Xty - XtX %*% y_acc) / n
    beta_next <- prox_oscar(y_acc - t_step * grad, t_step * lambda1, t_step * lambda2)
    
    if (sum((beta_next - beta)^2) < tol^2) {
      beta <- beta_next
      break
    }
    
    t_next <- (1 + sqrt(1 + 4 * t_acc^2)) / 2
    y_acc <- beta_next + ((t_acc - 1) / t_next) * (beta_next - beta)
    
    beta_prev <- beta
    beta <- beta_next
    t_acc <- t_next
  }
  
  return(beta)
}

# 5-fold Cross-Validation for OSCAR (Highly Optimized)
cv_oscar <- function(X, y, k_folds = 5, n_lambda1 = 5, n_lambda2 = 5) {
  n <- nrow(X)
  p <- ncol(X)
  
  Xty_full <- t(X) %*% y
  lambda1_max <- max(abs(Xty_full)) / n
  lambda1_grid <- seq(lambda1_max * 0.001, lambda1_max * 0.3, length.out = n_lambda1)
  lambda2_grid <- seq(0, lambda1_max * 0.05, length.out = n_lambda2)
  
  folds <- sample(rep(1:k_folds, length.out = n))
  cv_errors <- matrix(0, n_lambda1, n_lambda2)
  
  for (f in 1:k_folds) {
    X_train <- X[folds != f, , drop = FALSE]
    y_train <- y[folds != f]
    X_val <- X[folds == f, , drop = FALSE]
    y_val <- y[folds == f]
    
    # Precompute matrices for the training fold
    XtX_tr <- t(X_train) %*% X_train
    Xty_tr <- t(X_train) %*% y_train
    L_val <- max(eigen(XtX_tr, only.values = TRUE)$values)
    n_tr <- nrow(X_train)
    
    for (i in 1:n_lambda1) {
      for (j in 1:n_lambda2) {
        b <- fit_oscar_fixed(XtX_tr, Xty_tr, lambda1_grid[i], lambda2_grid[j], L_val / n_tr, n_tr, max_iter = 200)
        pred <- X_val %*% b
        cv_errors[i, j] <- cv_errors[i, j] + sum((y_val - pred)^2)
      }
    }
  }
  
  cv_errors <- cv_errors / n
  opt_idx <- which(cv_errors == min(cv_errors), arr.ind = TRUE)[1, ]
  return(list(
    lambda1 = lambda1_grid[opt_idx[1]],
    lambda2 = lambda2_grid[opt_idx[2]]
  ))
}

fit_oscar <- function(X, y, k_folds = 5) {
  opts <- cv_oscar(X, y, k_folds = k_folds)
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  L <- max(eigen(XtX, only.values = TRUE)$values)
  n_samples <- nrow(X)
  b <- fit_oscar_fixed(XtX, Xty, opts$lambda1, opts$lambda2, L / n_samples, n_samples)
  names(b) <- colnames(X)
  return(b)
}

# -------------------------------------------------------------
# 3. Bayesian Lasso (Gibbs Sampler)
# -------------------------------------------------------------
if (!exists("rmvnorm_robust")) {
  source("R/brgl.R")
}

BayesLasso <- function(X, y,
                       max_steps = 5000,
                       burn_in = 1000,
                       thin = 1,
                       a = 1.0,
                       b = 1.0,
                       seed = 1234) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  
  # Initialize
  lambda_sq <- rgamma(1, shape = a, rate = b)
  sigma2 <- var(y)
  tau_sq <- rexp(p, rate = lambda_sq / 2)
  beta <- numeric(p)
  
  n_saved <- floor((max_steps - burn_in) / thin)
  beta_samples <- matrix(0, nrow = n_saved, ncol = p)
  sigma2_samples <- numeric(n_saved)
  lambda_samples <- numeric(n_saved)
  
  save_idx <- 1
  
  for (step in 1:max_steps) {
    # 1. Update beta (block update)
    invD <- diag(1 / tau_sq, p)
    invA <- solve(XtX + invD)
    mu_be <- as.vector(invA %*% Xty)
    cov_be <- sigma2 * invA
    beta <- as.vector(rmvnorm_robust(1, mu_be, cov_be))
    
    # 2. Update tau_sq
    mu_tau <- sqrt(lambda_sq * sigma2 / (beta^2 + 1e-15))
    inv_tau_sq <- rinvgauss_vectorized(mu_tau, lambda_sq)
    tau_sq <- 1 / inv_tau_sq
    
    # 3. Update sigma^2
    shape_sig <- (n + p - 1) / 2
    resid_ss <- sum((y - X %*% beta)^2)
    prior_ss <- sum(beta^2 / tau_sq)
    scale_sig <- (resid_ss + prior_ss) / 2
    sigma2 <- 1 / rgamma(1, shape = shape_sig, rate = scale_sig)
    
    # 4. Update lambda
    shape_lam <- p + a
    scale_lam <- sum(tau_sq) / 2 + b
    lambda_sq <- rgamma(1, shape = shape_lam, rate = scale_lam)
    
    if (step > burn_in && (step - burn_in) %% thin == 0) {
      beta_samples[save_idx, ] <- beta
      sigma2_samples[save_idx] <- sigma2
      lambda_samples[save_idx] <- sqrt(lambda_sq)
      save_idx <- save_idx + 1
    }
  }
  
  beta_mean <- colMeans(beta_samples)
  names(beta_mean) <- colnames(X)
  
  return(list(
    beta = beta_mean,
    beta.post = beta_samples,
    sigma2.post = sigma2_samples,
    lambda.post = lambda_samples
  ))
}

# -------------------------------------------------------------
# 4. Bayesian Elastic Net (Gibbs Sampler)
# -------------------------------------------------------------
BayesElasticNet <- function(X, y,
                            max_steps = 5000,
                            burn_in = 1000,
                            thin = 1,
                            a1 = 1.0,
                            b1 = 1.0,
                            a2 = 1.0,
                            b2 = 0.1,
                            seed = 1234) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  
  # Initialize
  lambda1_sq <- rgamma(1, shape = a1, rate = b1)
  lambda2 <- rgamma(1, shape = a2, rate = b2)
  sigma2 <- var(y)
  tau_sq <- rexp(p, rate = lambda1_sq / 2)
  beta <- numeric(p)
  
  n_saved <- floor((max_steps - burn_in) / thin)
  beta_samples <- matrix(0, nrow = n_saved, ncol = p)
  sigma2_samples <- numeric(n_saved)
  lambda1_samples <- numeric(n_saved)
  lambda2_samples <- numeric(n_saved)
  
  save_idx <- 1
  
  for (step in 1:max_steps) {
    # 1. Update beta (block update)
    invD_star <- diag(1 / tau_sq + lambda2, p)
    invA <- solve(XtX + invD_star)
    mu_be <- as.vector(invA %*% Xty)
    cov_be <- sigma2 * invA
    beta <- as.vector(rmvnorm_robust(1, mu_be, cov_be))
    
    # 2. Update tau_sq
    mu_tau <- sqrt(lambda1_sq * sigma2 / (beta^2 + 1e-15))
    inv_tau_sq <- rinvgauss_vectorized(mu_tau, lambda1_sq)
    tau_sq <- 1 / inv_tau_sq
    
    # 3. Update sigma^2
    shape_sig <- (n + p - 1) / 2
    resid_ss <- sum((y - X %*% beta)^2)
    prior_ss <- sum(beta^2 * (1 / tau_sq + lambda2))
    scale_sig <- (resid_ss + prior_ss) / 2
    sigma2 <- 1 / rgamma(1, shape = shape_sig, rate = scale_sig)
    
    # 4. Update lambda1^2
    shape_lam1 <- p + a1
    scale_lam1 <- sum(tau_sq) / 2 + b1
    lambda1_sq <- rgamma(1, shape = shape_lam1, rate = scale_lam1)
    
    # 5. Update lambda2 via Metropolis-Hastings
    lambda2_prop <- rnorm(1, mean = lambda2, sd = 0.1)
    if (lambda2_prop > 0) {
      log_target_old <- (a2 - 1) * log(lambda2) - b2 * lambda2 + 0.5 * sum(log(1/tau_sq + lambda2)) - (lambda2 / (2 * sigma2)) * sum(beta^2)
      log_target_new <- (a2 - 1) * log(lambda2_prop) - b2 * lambda2_prop + 0.5 * sum(log(1/tau_sq + lambda2_prop)) - (lambda2_prop / (2 * sigma2)) * sum(beta^2)
      if (log(runif(1)) <= log_target_new - log_target_old) {
        lambda2 <- lambda2_prop
      }
    }
    
    if (step > burn_in && (step - burn_in) %% thin == 0) {
      beta_samples[save_idx, ] <- beta
      sigma2_samples[save_idx] <- sigma2
      lambda1_samples[save_idx] <- sqrt(lambda1_sq)
      lambda2_samples[save_idx] <- lambda2
      save_idx <- save_idx + 1
    }
  }
  
  beta_mean <- colMeans(beta_samples)
  names(beta_mean) <- colnames(X)
  
  return(list(
    beta = beta_mean,
    beta.post = beta_samples,
    sigma2.post = sigma2_samples,
    lambda1.post = lambda1_samples,
    lambda2.post = lambda2_samples
  ))
}
