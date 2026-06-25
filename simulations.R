# Parallelized Simulation Studies for BRGL and Competing Methods
# Paper Section 5.1

source("R/brgl.R")
source("R/competing_methods.R")
library(glmnet)
library(parallel)

# Number of replications (set to 100 for paper-matching results, or smaller e.g. 5 for testing)
n_reps <- 10

# Number of cores to use
n_cores <- 6

# MCMC settings (reproducing the 2000 iterations mentioned in the paper)
max_steps <- 2000
burn_in <- 500
thin <- 1

set.seed(1234)

# Define methods
methods_list <- c("BRGL", "Lasso", "EN", "OSCAR", "BLasso", "BEN")

# Helper function to generate X and y
generate_data <- function(scenario, n_train, n_test, p, sigma, seed) {
  set.seed(seed)
  n <- n_train + n_test
  
  if (scenario %in% c(1, 2, 3)) {
    # Autoregressive correlation: corr(x_i, x_j) = 0.7^|i-j|
    Sigma <- 0.7^abs(outer(1:p, 1:p, "-"))
    U <- chol(Sigma)
    X <- matrix(rnorm(n * p), n, p) %*% U
    
    if (scenario == 1) {
      beta <- c(3, 2, 1.5, rep(0, p - 3))
    } else if (scenario == 2) {
      beta <- c(3, 0, 0, 1.5, 0, 0, 0, 2)
    } else {
      beta <- rep(0.85, p)
    }
    
  } else if (scenario == 4) {
    # Compound symmetry: corr(x_i, x_j) = 0.5
    Sigma <- matrix(0.5, p, p)
    diag(Sigma) <- 1.0
    U <- chol(Sigma)
    X <- matrix(rnorm(n * p), n, p) %*% U
    beta <- c(rep(0, 10), rep(2, 10), rep(0, 10), rep(2, 10))
    
  } else if (scenario == 5) {
    # Grouping structure from Zou and Hastie (2005)
    Z1 <- rnorm(n)
    Z2 <- rnorm(n)
    Z3 <- rnorm(n)
    X <- matrix(0, n, p)
    for (i in 1:5) X[, i] <- Z1 + rnorm(n, sd = 0.4)
    for (i in 6:10) X[, i] <- Z2 + rnorm(n, sd = 0.4)
    for (i in 11:15) X[, i] <- Z3 + rnorm(n, sd = 0.4)
    for (i in 16:p) X[, i] <- rnorm(n)
    beta <- c(rep(3, 15), rep(0, p - 15))
  }
  
  # Standardize predictors
  X_mean <- colMeans(X)
  X_sd <- apply(X, 2, sd)
  X_std <- scale(X, center = X_mean, scale = X_sd)
  
  # Generate response y = X*beta + e (using raw X)
  eps <- rnorm(n, sd = sigma)
  y <- X %*% beta + eps
  y_mean <- mean(y[1:n_train])
  y_ctr <- y - y_mean
  
  X_train <- X_std[1:n_train, ]
  y_train <- y_ctr[1:n_train]
  X_test <- X_std[(n_train + 1):n, ]
  y_test <- y_ctr[(n_train + 1):n]
  
  return(list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    beta = beta,
    true_active = (beta != 0)
  ))
}

results_mse <- list()
results_metrics <- list()

for (scen in 1:5) {
  cat(sprintf("\n=== Running Scenario %d (n_reps = %d, cores = %d) ===\n", scen, n_reps, n_cores))
  
  if (scen %in% c(1, 2, 3)) {
    n_train <- 20; n_test <- 200; p <- 8; sigma <- 3
  } else {
    n_train <- 100; n_test <- 400; p <- 40; sigma <- 15
  }
  
  # Define the parallel job function for one replication
  run_replication <- function(r) {
    rep_seed <- 1234 + r
    data <- generate_data(scen, n_train, n_test, p, sigma, rep_seed)
    true_active <- data$true_active
    
    rep_mse <- numeric(length(methods_list))
    names(rep_mse) <- methods_list
    
    rep_time <- numeric(length(methods_list))
    names(rep_time) <- methods_list
    
    rep_metrics <- list()
    
    for (m in methods_list) {
      t_start <- proc.time()
      
      beta_est <- NULL
      beta_post_samples <- NULL
      
      # Fit model
      if (m == "BRGL") {
        fit <- brgl_mcmc(data$X_train, data$y_train, max_steps = max_steps, burn_in = burn_in, thin = thin, seed = rep_seed)
        beta_est <- fit$beta
        beta_post_samples <- fit$beta.post
      } else if (m == "Lasso") {
        beta_est <- fit_lasso(data$X_train, data$y_train)
      } else if (m == "EN") {
        beta_est <- fit_en(data$X_train, data$y_train)
      } else if (m == "OSCAR") {
        beta_est <- fit_oscar(data$X_train, data$y_train)
      } else if (m == "BLasso") {
        fit <- BayesLasso(data$X_train, data$y_train, max_steps = max_steps, burn_in = burn_in, thin = thin, seed = rep_seed)
        beta_est <- fit$beta
        beta_post_samples <- fit$beta.post
      } else if (m == "BEN") {
        fit <- BayesElasticNet(data$X_train, data$y_train, max_steps = max_steps, burn_in = burn_in, thin = thin, seed = rep_seed)
        beta_est <- fit$beta
        beta_post_samples <- fit$beta.post
      }
      
      t_end <- proc.time()
      rep_time[m] <- (t_end - t_start)["elapsed"]
      
      # Test MSE
      pred <- data$X_test %*% beta_est
      rep_mse[m] <- mean((data$y_test - pred)^2)
      
      # Variable selection (SNC or Non-zero)
      if (!is.null(beta_post_samples)) {
        post_vars <- apply(beta_post_samples, 2, var)
        post_probs <- colMeans(sweep(abs(beta_post_samples), 2, sqrt(post_vars), ">"))
        selected <- post_probs > 0.5
      } else {
        selected <- (beta_est != 0)
      }
      
      # Metrics
      tp <- sum(selected & true_active)
      fp <- sum(selected & (!true_active))
      tn <- sum(!selected & (!true_active))
      fn <- sum(!selected & true_active)
      
      tpr <- if (sum(true_active) > 0) tp / sum(true_active) else NA
      tnr <- if (sum(!true_active) > 0) tn / sum(!true_active) else NA
      ppv <- if (sum(selected) > 0) tp / sum(selected) else NA
      npv <- if (sum(!selected) > 0) tn / sum(!selected) else NA
      
      rep_metrics[[m]] <- c(TPR = tpr, TNR = tnr, PPV = ppv, NPV = npv)
    }
    
    return(list(mse = rep_mse, time = rep_time, metrics = rep_metrics))
  }
  
  # Run parallel replications
  cat("  Running replications in parallel...\n")
  t0 <- proc.time()
  results_list <- mclapply(1:n_reps, run_replication, mc.cores = n_cores)
  t1 <- proc.time()
  cat(sprintf("  Completed in %.2f seconds.\n", (t1 - t0)["elapsed"]))
  
  # Check for errors in parallel execution
  errors <- sapply(results_list, inherits, "try-error")
  if (any(errors)) {
    stop("Error occurred in one or more parallel processes.")
  }
  
  # Format results
  mse_df <- matrix(0, n_reps, length(methods_list))
  colnames(mse_df) <- methods_list
  
  time_df <- matrix(0, n_reps, length(methods_list))
  colnames(time_df) <- methods_list
  
  scen_metrics <- lapply(methods_list, function(m) {
    matrix(0, n_reps, 4, dimnames = list(NULL, c("TPR", "TNR", "PPV", "NPV")))
  })
  names(scen_metrics) <- methods_list
  
  for (r in 1:n_reps) {
    res <- results_list[[r]]
    for (m in methods_list) {
      mse_df[r, m] <- res$mse[m]
      time_df[r, m] <- res$time[m]
      for (met in c("TPR", "TNR", "PPV", "NPV")) {
        scen_metrics[[m]][r, met] <- res$metrics[[m]][met]
      }
    }
  }
  
  # Save raw results
  write.csv(mse_df, sprintf("results/scenario_%d_mse.csv", scen), row.names = FALSE)
  write.csv(time_df, sprintf("results/scenario_%d_time.csv", scen), row.names = FALSE)
  
  # Format Table 3: Test MSE percentiles and running times
  tbl3_scen <- matrix(0, length(methods_list), 4, dimnames = list(methods_list, c("10th Pct", "50th Pct", "90th Pct", "Median Time (s)")))
  for (m in methods_list) {
    pcts <- quantile(mse_df[, m], probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
    tbl3_scen[m, 1:3] <- pcts
    tbl3_scen[m, 4] <- median(time_df[, m], na.rm = TRUE)
  }
  write.csv(tbl3_scen, sprintf("results/table3_scenario_%d.csv", scen))
  results_mse[[scen]] <- tbl3_scen
  
  # Format Table 4: Operating characteristics
  tbl4_scen <- matrix(0, length(methods_list), 4, dimnames = list(methods_list, c("TPR", "TNR", "PPV", "NPV")))
  for (m in methods_list) {
    tbl4_scen[m, ] <- apply(scen_metrics[[m]], 2, median, na.rm = TRUE) * 100
  }
  write.csv(tbl4_scen, sprintf("results/table4_scenario_%d.csv", scen))
  results_metrics[[scen]] <- tbl4_scen
}

# Compile and print tables
cat("\n\n=========================================\n")
cat("SUMMARY OF RESULTS (Table 3: Test MSE Medians)\n")
cat("=========================================\n")
for (scen in 1:5) {
  cat(sprintf("\nScenario %d:\n", scen))
  print(round(results_mse[[scen]][, 1:3], 3))
}

cat("\n\n=========================================\n")
cat("SUMMARY OF RESULTS (Table 4: Selection Metrics Medians)\n")
cat("=========================================\n")
for (scen in 1:5) {
  cat(sprintf("\nScenario %d:\n", scen))
  print(round(results_metrics[[scen]], 1))
}
