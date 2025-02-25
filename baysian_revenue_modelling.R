# Author: Hayden Hedman
# Project: Bayesian MCMC Analysis for E-Commerce Revenue Modeling
# Date: 2025-02-06
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION:
## This model performs a Bayesian statistical analysis using Markov Chain Monte Carlo (MCMC) 
## to estimate how customer engagement (e.g., website visits, ad clicks) influences daily e-commerce revenue.
## The model is built using rstan, which provides efficient sampling using Hamiltonian Monte Carlo (HMC).
##
## Techniques applied:
## - Hierarchical Bayesian modeling using Stan.
## - MCMC sampling with multiple chains for robust inference.
## - Posterior summary tables and credible intervals for interpretation.
## - Graphical visualization of the data and MCMC results.

# OBJECTIVE:
## - Estimate the impact of customer engagement on e-commerce revenue.
## - Quantify uncertainty around model parameters using credible intervals.
## - Provide an  Bayesian summary output
## - Generate posterior predictive distributions for revenue.

# Hypothesis:
## - H0: Customer engagement (website visits, ad clicks) has no significant impact on daily revenue.
## - H1: Higher customer engagement significantly increases e-commerce revenue.
# ----------------------------------------------------------------------------
# Install required packages if not already installed
if (!require(rstan)) install.packages("rstan", dependencies=TRUE)
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(dplyr)) install.packages("dplyr")
if (!require(knitr)) install.packages("knitr")

# Load libraries
library(rstan)
library(ggplot2)
library(dplyr)
library(knitr)

# Set Stan options for faster compilation
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


# Set seed for reproducibility
set.seed(42)

# Generate synthetic data for 100 days
n_days <- 100
customer_engagement <- rnorm(n_days, mean = 500, sd = 100)  # Website visits or ad clicks
true_alpha <- 200  # Base revenue
true_beta <- 0.5   # Effect of engagement on revenue
true_sigma <- 50   # Revenue noise

# Generate revenue based on engagement + noise
revenue <- true_alpha + true_beta * customer_engagement + rnorm(n_days, 0, true_sigma)

# Store in a dataframe
ecom_data <- data.frame(day = 1:n_days, engagement = customer_engagement, revenue = revenue)

# Visualize the data
ggplot(ecom_data, aes(x = engagement, y = revenue)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Synthetic E-Commerce Revenue vs. Customer Engagement",
       x = "Customer Engagement (Website Visits / Ad Clicks)",
       y = "Daily Revenue ($)") +
  theme_minimal()


# Write the Stan model
stan_model_code <- "
data {
  int<lower=0> N;            // Number of observations
  vector[N] engagement;      // Predictor (customer engagement)
  vector[N] revenue;         // Response (daily revenue)
}

parameters {
  real alpha;                // Intercept (base revenue)
  real beta;                 // Effect of engagement on revenue
  real<lower=0> sigma;       // Standard deviation of noise
}

model {
  revenue ~ normal(alpha + beta * engagement, sigma); // Likelihood
}
"


# Prepare data for Stan
stan_data <- list(
  N = n_days,
  engagement = ecom_data$engagement,
  revenue = ecom_data$revenue
)

# Compile the model
stan_fit <- stan(model_code = stan_model_code, data = stan_data, 
                 iter = 2000, chains = 4, warmup = 1000, thin = 1)

# Print summary of results
print(stan_fit, pars = c("alpha", "beta", "sigma"))


# Extract samples from posterior
posterior_samples <- extract(stan_fit)

# Compute summary statistics
summary_table <- data.frame(
  Parameter = c("Intercept (alpha)", "Effect of Engagement (beta)", "Sigma (Noise)"),
  Mean = c(mean(posterior_samples$alpha), mean(posterior_samples$beta), mean(posterior_samples$sigma)),
  SD = c(sd(posterior_samples$alpha), sd(posterior_samples$beta), sd(posterior_samples$sigma)),
  `2.5%` = c(quantile(posterior_samples$alpha, 0.025),
             quantile(posterior_samples$beta, 0.025),
             quantile(posterior_samples$sigma, 0.025)),
  `97.5%` = c(quantile(posterior_samples$alpha, 0.975),
              quantile(posterior_samples$beta, 0.975),
              quantile(posterior_samples$sigma, 0.975))
)

# Print formatted summary table
cat("\n### Bayesian Model Summary Table ###\n")
kable(summary_table, format = "markdown")
