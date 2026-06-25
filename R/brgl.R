# Core Implementation of Bayesian Regularization via Graph Laplacian (BRGL)
# Paper: Fei Liu et al. (2014)

# -------------------------------------------------------------
# 1. Vectorized Robust Inverse Gaussian Sampler
# -------------------------------------------------------------
# Samples from IG(mu, lambda). If mu is extremely large (e.g. when beta is near 0),
# it converges to an Inverse Gamma(0.5, lambda / 2) distribution.
rinvgauss_vectorized <- function(mu, lambda) {
  n <- length(mu)
  out <- numeric(n)
  
  # Identify indices where mu is extremely large or infinite
  idx_large <- which(is.infinite(mu) | is.na(mu) | is.nan(mu) | mu > 1e8)
  idx_normal <- which(!is.infinite(mu) & !is.na(mu) & !is.nan(mu) & mu <= 1e8)
  
  if (length(idx_large) > 0) {
    lambdas_large <- if (length(lambda) == 1) rep(lambda, length(idx_large)) else lambda[idx_large]
    g <- rgamma(length(idx_large), shape = 0.5, rate = lambdas_large / 2)
    out[idx_large] <- 1 / g
  }
  
  if (length(idx_normal) > 0) {
    mu_n <- mu[idx_normal]
    lambda_n <- if (length(lambda) == 1) rep(lambda, length(idx_normal)) else lambda[idx_normal]
    
    # Standard Michael, Schucany, and Haas (1976) algorithm
    v <- rnorm(length(idx_normal))
    y <- v^2
    mu_sq <- mu_n^2
    temp <- sqrt(4 * mu_n * lambda_n * y + mu_sq * y^2)
    x1 <- mu_n + (mu_sq * y) / (2 * lambda_n) - (mu_n / (2 * lambda_n)) * temp
    
    # Handle potential numerical underflow or NaNs
    invalid <- is.na(x1) | x1 <= 0
    x1[invalid] <- mu_n[invalid]
    
    u <- runif(length(idx_normal))
    choose_x1 <- u <= mu_n / (mu_n + x1)
    
    out[idx_normal] <- ifelse(choose_x1, x1, mu_sq / x1)
  }
  
  return(out)
}

# -------------------------------------------------------------
# 2. Robust Multivariate Normal Sampler
# -------------------------------------------------------------
# Samples from N(mu, Sigma). Uses eigen decomposition fallback if Cholesky fails.
rmvnorm_robust <- function(n, mu, Sigma) {
  p <- length(mu)
  U <- tryCatch({
    chol(Sigma)
  }, error = function(e) {
    # Fallback to eigen decomposition for semi-definite matrices
    eg <- eigen(Sigma, symmetric = TRUE)
    t(eg$vectors %*% diag(sqrt(pmax(eg$values, 0)), p))
  })
  z <- matrix(rnorm(n * p), p, n)
  out <- t(U) %*% z + mu
  return(t(out))
}

# -------------------------------------------------------------
# 3. Core BRGL Gibbs Sampler
# -------------------------------------------------------------
brgl_mcmc <- function(X, y,
                      max_steps = 5000,
                      burn_in = 1000,
                      thin = 1,
                      hr = 1.0,      # Hyperprior shape for r
                      gr = 0.01,     # Hyperprior rate for r
                      ga = 0.01,     # Hyperprior rate for a
                      gb = 0.01,     # Hyperprior rate for b
                      seed = 1234) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  yty <- sum(y^2)
  
  # Index vectors for lower triangular part (off-diagonals)
  lt_idx <- which(lower.tri(matrix(0, p, p)), arr.ind = TRUE)
  i_cols <- lt_idx[, 1]
  j_cols <- lt_idx[, 2]
  n_off <- nrow(lt_idx)
  
  # Initialization
  sigma2 <- var(y)
  beta <- as.vector(solve(XtX + diag(0.1, p)) %*% Xty)
  r <- 1.0
  a <- 1.0
  b <- 1.0
  
  c_vec <- if (n_off > 0) sample(c(-1, 1), n_off, replace = TRUE) else numeric(0)
  eta_diag <- rep(1.0, p)
  eta_off <- if (n_off > 0) rep(1.0, n_off) else numeric(0)
  
  # Storage
  n_saved <- floor((max_steps - burn_in) / thin)
  beta_samples <- matrix(0, nrow = n_saved, ncol = p)
  sigma2_samples <- numeric(n_saved)
  r_samples <- numeric(n_saved)
  a_samples <- numeric(n_saved)
  b_samples <- numeric(n_saved)
  Lambda_post_sum <- matrix(0, p, p)
  
  # Construct initial Lambda
  lambda_mat <- matrix(0, p, p)
  diag(lambda_mat) <- eta_diag
  if (n_off > 0) {
    lambda_mat[lt_idx] <- c_vec * eta_off
    lambda_mat[lt_idx[, c(2, 1)]] <- c_vec * eta_off
  }
  
  Lambda <- lambda_mat
  diag(Lambda) <- 1 + diag(lambda_mat) + rowSums(abs(lambda_mat)) - diag(abs(lambda_mat))
  
  save_idx <- 1
  
  for (step in 1:max_steps) {
    # -------------------------------------------------------------
    # (i) Update sigma^2
    # -------------------------------------------------------------
    inv_mat <- solve(XtX + r * Lambda)
    mu_beta <- inv_mat %*% Xty
    rate_sig <- as.numeric(yty - t(Xty) %*% mu_beta) / 2
    rate_sig <- max(rate_sig, 1e-10) # numerical safety
    
    sigma2 <- 1 / rgamma(1, shape = n / 2, rate = rate_sig)
    sigma <- sqrt(sigma2)
    
    # -------------------------------------------------------------
    # (ii) Update beta
    # -------------------------------------------------------------
    cov_beta <- sigma2 * inv_mat
    beta <- as.vector(rmvnorm_robust(1, mu_beta, cov_beta))
    
    # -------------------------------------------------------------
    # (iii) Update signs c_ij (j < i)
    # -------------------------------------------------------------
    if (n_off > 0) {
      diff_abs <- abs(beta[i_cols] - beta[j_cols])
      sum_abs <- abs(beta[i_cols] + beta[j_cols])
      p_ij <- 1 / (1 + exp(-r * b * (diff_abs - sum_abs) / (2 * sigma)))
      p_ij[is.na(p_ij)] <- 0.5
      c_vec <- ifelse(runif(n_off) <= p_ij, 1, -1)
    }
    
    # -------------------------------------------------------------
    # (iv) Update η_ii and η_ij
    # -------------------------------------------------------------
    mu_diag <- (a * sigma) / (sqrt(r) * abs(beta) + 1e-15)
    eta_diag <- rinvgauss_vectorized(mu_diag, a^2)
    
    if (n_off > 0) {
      mu_off <- (b * sigma) / (sqrt(r) * abs(beta[i_cols] + c_vec * eta_off) + 1e-15) # fix typo to use correct beta grouping
      mu_off <- (b * sigma) / (sqrt(r) * abs(beta[i_cols] + c_vec * beta[j_cols]) + 1e-15)
      eta_off <- rinvgauss_vectorized(mu_off, b^2)
    }
    
    # -------------------------------------------------------------
    # (v) Set λ_ii and λ_ij, and construct Lambda
    # -------------------------------------------------------------
    lambda_mat <- matrix(0, p, p)
    diag(lambda_mat) <- eta_diag
    if (n_off > 0) {
      lambda_mat[lt_idx] <- c_vec * eta_off
      lambda_mat[lt_idx[, c(2, 1)]] <- c_vec * eta_off
    }
    
    Lambda <- lambda_mat
    diag(Lambda) <- 1 + diag(lambda_mat) + rowSums(abs(lambda_mat)) - diag(abs(lambda_mat))
    
    # -------------------------------------------------------------
    # (vi) Update hyperparameters r, a, b
    # -------------------------------------------------------------
    sum_beta_sq <- sum(beta^2)
    sum_beta_abs <- sum(abs(beta))
    sum_beta_grouped_abs <- if (n_off > 0) sum(abs(beta[i_cols] + c_vec * beta[j_cols])) else 0.0
    
    rate_r <- sum_beta_sq / (2 * sigma2) + (a * sum_beta_abs) / (2 * sigma) + (b * sum_beta_grouped_abs) / (2 * sigma) + gr
    r <- rgamma(1, shape = p / 2 + hr, rate = rate_r)
    r <- max(r, 1e-10)
    
    rate_a <- ga + (r * sum_beta_abs) / (2 * sigma)
    a <- rexp(1, rate = rate_a)
    
    rate_b <- gb + (r * sum_beta_grouped_abs) / (2 * sigma)
    b <- rexp(1, rate = rate_b)
    
    # -------------------------------------------------------------
    # Save samples after burn-in
    # -------------------------------------------------------------
    if (step > burn_in && (step - burn_in) %% thin == 0) {
      beta_samples[save_idx, ] <- beta
      sigma2_samples[save_idx] <- sigma2
      r_samples[save_idx] <- r
      a_samples[save_idx] <- a
      b_samples[save_idx] <- b
      Lambda_post_sum <- Lambda_post_sum + Lambda
      save_idx <- save_idx + 1
    }
  }
  
  beta_mean <- colMeans(beta_samples)
  sigma2_mean <- mean(sigma2_samples)
  r_mean <- mean(r_samples)
  a_mean <- mean(a_samples)
  b_mean <- mean(b_samples)
  Lambda_mean <- Lambda_post_sum / n_saved
  
  names(beta_mean) <- colnames(X)
  
  return(list(
    beta = beta_mean,
    sigma2 = sigma2_mean,
    r = r_mean,
    a = a_mean,
    b = b_mean,
    Lambda = Lambda_mean,
    beta.post = beta_samples,
    sigma2.post = sigma2_samples,
    r.post = r_samples,
    a.post = a_samples,
    b.post = b_samples
  ))
}
