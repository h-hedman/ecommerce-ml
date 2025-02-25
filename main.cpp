// Author: Hayden Hedman  
// Project: Profit Optimization Using Golden Ratio in E-Commerce Pricing  
// Date: 2025-02-25  
// --------------------------------------------------------------------------------------------------------------  
// DESCRIPTION:  
// This program determines the optimal product pricing for maximum profit using the Golden Section Search method.  
// It models demand as a linear function of price and calculates profit accordingly.  
// The algorithm iteratively refines price estimates to maximize revenue while considering cost constraints.  
//
// OBJECTIVE:  
// - Identify the optimal price that maximizes profit in an e-commerce setting.  
// - Understand how price sensitivity affects demand and revenue.  
// - Utilize the Golden Section Search algorithm to efficiently search for the best price.  
//
// HYPOTHESIS:  
// H0: Price does not significantly impact demand and profit.  
// H1: There exists an optimal price that maximizes profit by balancing price sensitivity and unit costs.  
// --------------------------------------------------------------------------------------------------------------
#include <iostream>
#include <cmath>
#include <iomanip>
// --------------------------------------------------------------------------------------------------------------
// ECONOMIC MODEL PARAMETERS  
// These constants define the market dynamics, including demand behavior and production cost.
const double MAXIMUM_DEMAND = 1000.0;  // Demand when the price is zero
const double PRICE_SENSITIVITY = 5.0;  // Rate at which demand decreases as price increases
const double UNIT_COST = 10.0;         // Fixed cost per unit produced
// --------------------------------------------------------------------------------------------------------------
// FUNCTION: computeDemand  
// PURPOSE: Computes the expected demand given a specific price based on a linear demand function.  
// PARAMETERS:  
// - price (double): The selling price of the product.  
// RETURNS:  
// - (double) Demand at the specified price (ensures non-negative demand).  
double computeDemand(double price) {
    double demand = MAXIMUM_DEMAND - PRICE_SENSITIVITY * price;
    return std::max(demand, 0.0);  // Ensure demand does not fall below zero
}
// --------------------------------------------------------------------------------------------------------------
// FUNCTION: computeProfit  
// PURPOSE: Computes the total profit at a given price.  
// PARAMETERS:  
// - price (double): The selling price of the product.  
// RETURNS:  
// - (double) Profit calculated as (price - cost) * demand.  
double computeProfit(double price) {
    double demand = computeDemand(price);
    return (price - UNIT_COST) * demand;
}
// --------------------------------------------------------------------------------------------------------------
// FUNCTION: goldenSectionSearch  
// PURPOSE: Implements the Golden Section Search algorithm to find the optimal price for maximum profit.  
// PARAMETERS:  
// - lowerBound (double): The minimum allowable price (must cover unit cost).  
// - upperBound (double): The maximum price where demand approaches zero.  
// - tolerance (double): The precision level for stopping the search (default = 1e-6).  
// RETURNS:  
// - (double) The price that maximizes profit.  
double goldenSectionSearch(double lowerBound, double upperBound, double tolerance = 1e-6) {
    const double GOLDEN_RATIO = (1 + std::sqrt(5)) / 2.0;  // Approx. 1.618 (φ)
    const double RESCALED_RATIO = 2 - GOLDEN_RATIO;        // 1 / φ^2

    // Initial points based on golden ratio
    double price1 = lowerBound + RESCALED_RATIO * (upperBound - lowerBound);
    double price2 = upperBound - RESCALED_RATIO * (upperBound - lowerBound);
    double profit1 = computeProfit(price1);
    double profit2 = computeProfit(price2);

    // Iteratively refine the search interval to converge to optimal price
    while (std::fabs(upperBound - lowerBound) > tolerance) {
        if (profit1 < profit2) {
            lowerBound = price1;
            price1 = price2;
            profit1 = profit2;
            price2 = upperBound - RESCALED_RATIO * (upperBound - lowerBound);
            profit2 = computeProfit(price2);
        } else {
            upperBound = price2;
            price2 = price1;
            profit2 = profit1;
            price1 = lowerBound + RESCALED_RATIO * (upperBound - lowerBound);
            profit1 = computeProfit(price1);
        }
    }
    
    return (lowerBound + upperBound) / 2;  // Return the midpoint as the optimal price
}
// --------------------------------------------------------------------------------------------------------------
// MAIN FUNCTION  
// PURPOSE: Defines the search boundaries and computes the optimal price and corresponding profit.  
// OUTPUT: Displays the optimal price and maximum profit to the console.  
int main() {
    // Define search interval for price optimization:
    double minimumPrice = UNIT_COST;                        // The lowest possible price (ensuring no loss)
    double maximumPrice = MAXIMUM_DEMAND / PRICE_SENSITIVITY; // The price at which demand reaches zero

    // Compute the optimal price and corresponding profit
    double optimalPrice = goldenSectionSearch(minimumPrice, maximumPrice);
    double optimalProfit = computeProfit(optimalPrice);

    // Display results with precision formatting
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Optimal Price: $" << optimalPrice << std::endl;
    std::cout << "Maximum Profit: $" << optimalProfit << std::endl;

    return 0;
}
// --------------------------------------------------------------------------------------------------------------
