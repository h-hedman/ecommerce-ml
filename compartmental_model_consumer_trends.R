# Author: Hayden Hedman
# Project: Compartmental Revenue Modeling for E-Commerce (Visitors → Engaged → Purchasers)
# Date: 2025-02-18
# --------------------------------------------------------------------------------------------------------------
# DESCRIPTION:
## This model simulates the flow of users in an e-commerce setting using a compartmental system:
## - Visitors (V) → Engaged Users (E) → Purchasers (P)
## It leverages **Ordinary Differential Equations (ODEs)** to represent the transitions between customer states.
## 
## The model helps **analyze customer engagement and conversion rates** over time by solving differential equations 
## and visualizing the dynamics of user flow.
##
## Key techniques applied:
## - **System of ODEs** to model state transitions.
## - **Numerical ODE solver (`deSolve`)** for accurate forecasting.
## - **`ggplot2` visualization** for clear data interpretation.
## - **Summary table (`kable`)** to display key transition metrics.

# OBJECTIVE:
## - Simulate and evaluate the transition of users from **Visitors → Engaged → Purchasers**.
## - Quantify the effectiveness of engagement and purchase rates (`beta`, `gamma`).
## - Generate **actionable insights** into customer retention and revenue conversion dynamics.

# Hypothesis:
## - **H0 (Null Hypothesis):** The engagement rate (V → E) and purchase rate (E → P) do not significantly affect 
##   the number of purchasers over time.
## - **H1 (Alternative Hypothesis):** Higher engagement and purchase rates significantly increase 
##   the number of users who convert to purchases.
# --------------------------------------------------------------------------------------------------------------
# INSTALL & LOAD REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------------
if (!require(deSolve)) install.packages("deSolve")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(viridis)) install.packages("viridis")
if (!require(reshape2)) install.packages("reshape2")
if (!require(dplyr)) install.packages("dplyr")
if (!require(knitr)) install.packages("knitr")

library(deSolve)
library(ggplot2)
library(viridis)
library(reshape2)
library(dplyr)
library(knitr)

# --------------------------------------------------------------------------------------------------------------
# DEFINE COMPARTMENTAL MODEL (VEP: Visitors → Engaged → Purchasers)
# --------------------------------------------------------------------------------------------------------------
vep_model <- function(time, state, parameters) {
  V <- state[1]  # Visitors
  E <- state[2]  # Engaged Users
  P <- state[3]  # Purchasers
  
  beta <- parameters["beta"]   # Engagement Rate (V -> E)
  gamma <- parameters["gamma"] # Purchase Rate (E -> P)
  
  dV <- -beta * V              # Visitors converting to engaged users
  dE <- beta * V - gamma * E   # Engaged users transitioning to purchases
  dP <- gamma * E              # Purchasers growing over time
  
  list(c(dV, dE, dP))
}

# --------------------------------------------------------------------------------------------------------------
# INITIAL CONDITIONS & PARAMETERS
# --------------------------------------------------------------------------------------------------------------

# Initial state: 1000 visitors, no engaged or purchasers
initial_state <- c(V = 1000, E = 0, P = 0)

# Define simulation time in minutes (0 to 240 minutes, step of 10)
time <- seq(0, 240, by = 10)  # 4-hour window

# Model parameters: Engagement & purchase rates
parameters <- c(beta = 0.02, gamma = 0.015)  # Higher engagement & purchase speed

# --------------------------------------------------------------------------------------------------------------
# SOLVE THE ODE SYSTEM
# --------------------------------------------------------------------------------------------------------------

# Solve the system using ode function
output <- ode(y = initial_state, times = time, func = vep_model, parms = parameters)

# Convert output to a data frame
output_df <- as.data.frame(output)

# Reshape data for ggplot
output_long <- melt(output_df, id.vars = "time", variable.name = "State", value.name = "Users")

# Rename the legend labels for clarity
output_long$State <- factor(output_long$State, 
                            levels = c("V", "E", "P"), 
                            labels = c("Visitors", "Engaged", "Purchasers"))

# --------------------------------------------------------------------------------------------------------------
# PLOT RESULTS WITH GGPLOT2
# --------------------------------------------------------------------------------------------------------------
ggplot(output_long, aes(x = time, y = Users, color = State)) +
  geom_line(linewidth = 1.5) +  # Replaced `size` with `linewidth` for ggplot2 v3.4+
  scale_color_viridis_d() +  # Use viridis color scale for better visibility
  labs(title = "VEP Consumer Flow Model", 
       y = "Number of Users", 
       x = "Time (Minutes)", 
       color = "Customer State") +  # Legend title
  theme_minimal(base_size = 12) +  # Use a clean, modern theme
  theme(legend.position = "top",  
        legend.title = element_text(size = 14, face = "bold"),  
        legend.text = element_text(size = 12),  
        axis.text = element_text(size = 12),  
        axis.title = element_text(size = 14),  
        plot.title = element_text(size = 16, face = "bold")) +  
  geom_point(size = 2, alpha = 0.6) +  # Add points for visibility
  geom_vline(xintercept = 120, linetype = "dashed", color = "gray") +  # Reference line at 2 hours
  annotate("text", x = 130, y = max(output_df$V), label = "2-Hour Mark", size = 5, hjust = 0)

# --------------------------------------------------------------------------------------------------------------
# SUMMARY TABLE OF KEY ESTIMATES
# --------------------------------------------------------------------------------------------------------------
# Extract key values: Initial and Final counts
summary_table <- data.frame(
  State = c("Visitors", "Engaged", "Purchasers"),
  Initial = c(initial_state["V"], initial_state["E"], initial_state["P"]),
  Final = c(tail(output_df$V, 1), tail(output_df$E, 1), tail(output_df$P, 1))
)

# Calculate percentage change for each state
summary_table$Change_Percentage <- round(((summary_table$Final - summary_table$Initial) / summary_table$Initial) * 100, 2)

# Print summary table
cat("\n### Summary of Key Estimates ###\n")
kable(summary_table, format = "markdown")
# --------------------------------------------------------------------------------------------------------------

