# Bayesian Regularization via Graph Laplacian (BRGL)

An R implementation of the **Bayesian Regularization via Graph Laplacian (BRGL)** method for variable selection and grouping in linear regression, as proposed by Fei Liu, Sounak Chakraborty, Fan Li, Yan Liu, and Aurelie C. Lozano in *Bayesian Analysis* (2014).

This repository contains the core Gibbs sampler implementation, wrappers for competing methods, simulation scripts replicating the paper's scenarios, real data analysis scripts on company KPI metrics, and the generated figures and tables.

---

## 📖 Model Formulation

Consider the normal linear regression model:
$$y = X\beta + \epsilon, \quad \epsilon \sim N(0, \sigma^2 I_n)$$

To explicitly model the dependence and grouping structure between predictor variables without the prohibitive cost of inverting a covariance matrix, the BRGL method assigns a prior on the precision matrix $\Lambda$ using a generalized graph Laplacian:
$$\beta \mid \sigma^2 \sim N_p\left(0, \frac{\sigma^2}{r} \Lambda^{-1}\right)$$

where $\Lambda$ takes the form:
$$\Lambda = \begin{pmatrix} 
1 + \lambda_{11} + \sum_{j \neq 1} |\lambda_{1j}| & \lambda_{12} & \dots & \lambda_{1p} \\
\lambda_{21} & 1 + \lambda_{22} + \sum_{j \neq 2} |\lambda_{2j}| & \dots & \lambda_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\lambda_{p1} & \dots & \dots & 1 + \lambda_{pp} + \sum_{j \neq p} |\lambda_{pj}|
\end{pmatrix}$$

with $\lambda_{ij} = \lambda_{ji}$, $\lambda_{ii} > 0$, and hyperparameter $r \ge 0$. The prior distribution for the parameter vector $\lambda$ is defined as:
$$\pi(\lambda) \propto C_{a,b} |\Lambda|^{-1/2} \prod_{i=1}^p \lambda_{ii}^{-3/2} \exp\left(-\frac{a^2}{2\lambda_{ii}}\right) 1(\lambda_{ii} > 0) \prod_{j < i} |\lambda_{ij}|^{-3/2} \exp\left(-\frac{b^2}{2|\lambda_{ij}|}\right)$$

This specification cancels out the $|\Lambda|^{1/2}$ term in the likelihood, leading to a closed-form marginal prior for $\beta$ that encourages both sparsity and pairwise grouping (similar to Elastic Net and OSCAR):
$$\pi(\beta \mid c, r, a, b, \sigma^2) \propto (2\pi\sigma^2)^{-p/2} \exp \left\{ -\frac{1}{2\sigma^2} \left( r \sum_i \beta_i^2 + r a \sigma \sum_i |\beta_i| + r b \sigma \sum_{j < i} |\beta_i + c_{ij} \beta_j| \right) \right\}$$

---

## 🛠️ Gibbs Sampler Updates

Posterior inference is carried out using a Markov Chain Monte Carlo (MCMC) algorithm based on parameter augmentation (introducing $\eta_{ij} = |\lambda_{ij}|$ and $c_{ij} = \text{sign}(\lambda_{ij})$):

1. **Update $\sigma^2$** from its conditional Inverse-Gamma distribution:
   $$\sigma^2 \mid \lambda, D \sim \text{Inv-Gamma}\left( \frac{n}{2}, \frac{y' (I_n - X(X'X + r\Lambda)^{-1}X')y}{2} \right)$$

2. **Update $\beta$** from its conditional Multivariate Normal distribution:
   $$\beta \mid \sigma^2, \lambda, D \sim N_p(\mu_\beta, \Sigma_\beta)$$
   where $\mu_\beta = (X'X + r\Lambda)^{-1} X'y$ and $\Sigma_\beta = \sigma^2 (X'X + r\Lambda)^{-1}$.

3. **Update signs $c_{ij}$** (for $j < i$) independently from Bernoulli distributions:
   $$P(c_{ij} = 1 \mid \beta, \sigma^2) = p_{ij} = \left[ 1 + \exp \left\{ - \frac{r b (|\beta_i - \beta_j| - |\beta_i + \beta_j|)}{2\sigma} \right\} \right]^{-1}$$

4. **Update augmented parameters $\eta_{ii}$ and $\eta_{ij}$** from Inverse Gaussian distributions:
   $$\eta_{ii} \mid \beta, \sigma^2 \sim \text{Inv-Gaussian}\left( \frac{a\sigma}{\sqrt{r} |\beta_i|}, a^2 \right)$$
   $$\eta_{ij} \mid c, \beta, \sigma^2 \sim \text{Inv-Gaussian}\left( \frac{b\sigma}{\sqrt{r} |\beta_i + c_{ij} \beta_j|}, b^2 \right)$$

5. **Set $\lambda_{ii} = \eta_{ii}$ and $\lambda_{ij} = c_{ij} \eta_{ij}$**, and reconstruct $\Lambda$.

6. **Update hyperparameters $r, a, b$** from their respective full conditionals:
   $$r \mid a, b, c, \beta \sim \text{Gamma}\left( \frac{p}{2} + h_r, \frac{\sum_i \beta_i^2}{2\sigma^2} + \frac{a \sum_i |\beta_i|}{2\sigma} + \frac{b \sum_{j < i} |\beta_i + c_{ij} \beta_j|}{2\sigma} + g_r \right)$$
   $$a \mid r, \beta, \sigma \sim \text{Exp}\left( g_a + \frac{r \sum_i |\beta_i|}{2\sigma} \right)$$
   $$b \mid r, \beta, \sigma \sim \text{Exp}\left( g_b + \frac{r \sum_{j < i} |\beta_i + c_{ij} \beta_j|}{2\sigma} \right)$$

---

## 📂 Repository Structure

- **`R/brgl.R`**: Core implementation of the BRGL Gibbs sampler and its auxiliary sampling functions (`rinvgauss_vectorized`, `rmvnorm_robust`).
- **`R/competing_methods.R`**: Standard wrappers and custom solvers for:
  - **Lasso & Elastic Net** (using `glmnet` cross-validation)
  - **OSCAR** (via a custom FISTA + PAVA solver and cross-validation)
  - **Bayesian Lasso & Bayesian Elastic Net** (custom Gibbs samplers)
- **`simulations.R`**: Simulation runner for the 5 scenarios described in Section 5.1 of the paper (runs in parallel across CPU cores).
- **`real_data_analysis.R`**: Script simulating a company KPI dataset of size $n=120, p=60$ and generating all corresponding heatmaps, histograms, and network graphs.
- **`test_run.R`**: Validation script to verify the correctness of the code.
- **`paper.pdf`**: The original journal publication.
- **`figures/`**: Folder containing the generated plots.
- **`results/`**: Folder containing the generated tables and raw results.

---

## 🚀 How to Run the Scripts

Ensure you have R and the `glmnet`, `ggplot2`, and `gridExtra` packages installed.

1. **Verify code correctness (fast test):**
   ```bash
   Rscript test_run.R
   ```

2. **Run KPI Real Data Analysis and generate figures:**
   ```bash
   Rscript real_data_analysis.R
   ```
   This will output the following figures in the `figures/` directory:
   - `figure2.png`: L2 norm contribution of sorted coefficients.
   - `figure3.png`: Histogram of the off-diagonal entries in the estimated correlation matrix.
   - `figure4.png`: Side-by-side comparison of the sample correlation of $X$ and the estimated correlation of $\beta$.
   - `figure5.png`: Network graph of selected features showing correlation strengths.

3. **Run the 5 simulation studies:**
   ```bash
   Rscript simulations.R
   ```
   This script parallelizes the 100 replications for all 5 scenarios over 6 CPU cores, saving the resulting CSV files and displaying summary tables in the console.

---

## 📊 Summary of Simulation Results

*Note: The following tables are updated from the parallel simulation runs.*

### Table 3: Test Mean Squared Error (MSE) Percentiles (10th / 50th / 90th)
| Study | BRGL | Lasso | EN | OSCAR | BLasso | BEN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 10.6 / 12.3 / 20.4 | 9.5 / 12.9 / 16.8 | 10.2 / 11.7 / 17.1 | 9.7 / 12.0 / 19.8 | 9.5 / 12.1 / 21.1 | 11.4 / 14.8 / 19.8 |
| **2** | 10.4 / 13.2 / 15.5 | 10.1 / 12.6 / 15.4 | 10.0 / 12.2 / 15.8 | 9.8 / 12.4 / 14.1 | 10.1 / 12.4 / 15.9 | 10.2 / 13.6 / 16.0 |
| **3** | 9.6 / 10.8 / 13.5 | 10.4 / 14.1 / 16.1 | 9.2 / 12.9 / 15.0 | 8.8 / 10.7 / 14.1 | 10.2 / 12.0 / 13.5 | 8.9 / 10.6 / 14.3 |
| **4** | 242.7 / 265.9 / 327.8 | 255.9 / 277.5 / 346.2 | 246.5 / 270.8 / 339.8 | 254.7 / 289.1 / 355.8 | 266.0 / 312.9 / 374.1 | 249.5 / 283.2 / 331.7 |
| **5** | 241.9 / 286.7 / 358.6 | 254.1 / 266.7 / 336.5 | 244.0 / 263.4 / 330.5 | 244.1 / 270.7 / 349.3 | 255.1 / 307.9 / 386.7 | 237.4 / 282.2 / 364.8 |

### Table 4: Median Operating Characteristics (%)
| Study | Method | TPR | TNR | PPV | NPV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **BRGL** | **100.0** | **90.0** | **87.5** | **100.0** |
| | Lasso | 100.0 | 50.0 | 46.4 | 100.0 |
| | EN | 100.0 | 60.0 | 55.0 | 100.0 |
| | OSCAR | 100.0 | 30.0 | 37.5 | 100.0 |
| | BLasso | 100.0 | 100.0 | 100.0 | 100.0 |
| | BEN | 83.3 | 100.0 | 100.0 | 91.7 |
| **2** | **BRGL** | **66.7** | **90.0** | **83.3** | **83.3** |
| | Lasso | 100.0 | 60.0 | 55.0 | 100.0 |
| | EN | 100.0 | 50.0 | 55.0 | 100.0 |
| | OSCAR | 100.0 | 30.0 | 43.8 | 100.0 |
| | BLasso | 66.7 | 100.0 | 100.0 | 83.3 |
| | BEN | 66.7 | 90.0 | 83.3 | 81.7 |
| **3** | **BRGL** | **56.2** | **NA** | **100.0** | **0.0** |
| | Lasso | 75.0 | NA | 100.0 | 0.0 |
| | EN | 87.5 | NA | 100.0 | 0.0 |
| | OSCAR | 100.0 | NA | 100.0 | 0.0 |
| | BLasso | 37.5 | NA | 100.0 | 0.0 |
| | BEN | 50.0 | NA | 100.0 | 0.0 |
| **4** | **BRGL** | **47.5** | **80.0** | **70.7** | **59.6** |
| | Lasso | 67.5 | 70.0 | 70.0 | 70.0 |
| | EN | 70.0 | 60.0 | 65.0 | 70.4 |
| | OSCAR | 65.0 | 75.0 | 72.2 | 67.4 |
| | BLasso | 45.0 | 75.0 | 69.0 | 57.0 |
| | BEN | 40.0 | 82.5 | 71.8 | 56.9 |
| **5** | **BRGL** | **76.7** | **70.0** | **59.5** | **82.2** |
| | Lasso | 73.3 | 80.0 | 67.7 | 83.3 |
| | EN | 86.7 | 70.0 | 65.1 | 90.9 |
| | OSCAR | 73.3 | 78.0 | 65.6 | 82.1 |
| | BLasso | 46.7 | 66.0 | 45.2 | 67.3 |
| | BEN | 66.7 | 74.0 | 62.7 | 79.1 |

---

## 📈 Figures from KPI Data Analysis

### Figure 2: Coefficient Contribution to $L_2$ Norm
![Figure 2](figures/figure2.png)

### Figure 3: Histogram of Off-Diagonal Estimated Correlations
![Figure 3](figures/figure3.png)

### Figure 4: Sample Correlation Matrix of X vs. Estimated Correlation of Beta
![Figure 4](figures/figure4.png)

### Figure 5: Estimated Correlation Network Graph
![Figure 5](figures/figure5.png)
