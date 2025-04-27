import numpy as np
import random
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Function to apply Laplace mechanism for differential privacy
def laplace_mechanism(location_data, epsilon, sensitivity):
    """
    Add Laplace noise to the location data for differential privacy.
    :param location_data: Actual location data (e.g., coordinates).
    :param epsilon: Privacy budget (lower epsilon means more noise).
    :param sensitivity: Sensitivity of the query (e.g., the maximum change in data).
    :return: Noisy location data (differentially private).
    """
    noise = np.random.laplace(0, sensitivity / epsilon, size=location_data.shape)
    return location_data + noise

# Function to apply distortion-privacy mechanism
def distort_location(location_data, delta_distortion):
    """
    Apply distortion-privacy by introducing random distortion to the location data.
    :param location_data: Actual location data.
    :param delta_distortion: Maximum allowed distortion.
    :return: Distorted location data.
    """
    distorted_data = []
    for loc in location_data:
        distortion = random.uniform(-delta_distortion, delta_distortion)
        distorted_data.append(loc + distortion)
    return np.array(distorted_data)

# Function for adjusting data based on obfuscated location
def adjust_data_to_obfuscated_location(original_data, obfuscated_location):
    """
    Adjust sensing data based on the obfuscated location using linear regression.
    :param original_data: Original data (e.g., sensor readings).
    :param obfuscated_location: The obfuscated location.
    :return: Adjusted sensing data.
    """
    model = LinearRegression()
    model.fit(obfuscated_location.reshape(-1, 1), original_data)
    adjusted_data = model.predict(obfuscated_location.reshape(-1, 1))
    return adjusted_data

# Linear program to solve the optimal location obfuscation
def optimize_location_obfuscation(location_data, epsilon, delta_distortion):
    """
    Solve the linear program to find the optimal obfuscation function.
    :param location_data: Actual location data.
    :param epsilon: Privacy budget (differential privacy).
    :param delta_distortion: Maximum distortion allowed (distortion privacy).
    :return: Optimized obfuscated locations.
    """
    # Example of a simple linear program (minimizing obfuscation while maintaining privacy constraints)
    n = len(location_data)
    c = np.ones(n)  # Minimize obfuscation
    A = np.vstack([np.eye(n), -np.eye(n)])  # Ensure distortion is within delta bounds
    b = np.hstack([delta_distortion * np.ones(n), delta_distortion * np.ones(n)])
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method='highs')
    
    if result.success:
        return result.x  # Optimized obfuscated locations
    else:
        return None

# Uncertainty-aware inference algorithm using Gaussian Process Regression
def uncertainty_aware_inference(obfuscated_data, original_data):
    """
    Apply a Gaussian Process model to account for uncertainty in the inference.
    :param obfuscated_data: The data after obfuscation.
    :param original_data: The original sensing data for comparison.
    :return: Inferred data considering uncertainty.
    """
    kernel = RBF()
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(obfuscated_data.reshape(-1, 1), original_data)
    
    inferred_data, std_dev = model.predict(obfuscated_data.reshape(-1, 1), return_std=True)
    return inferred_data, std_dev

# Main function to simulate the privacy-enhanced MCS framework
def privacy_enhanced_mcs(location_data, original_data, epsilon, delta_distortion, sensitivity):
    """
    Main function to simulate the privacy-enhanced Sparse Mobile Crowdsensing framework.
    :param location_data: Actual location data.
    :param original_data: Original sensing data (e.g., traffic, environment data).
    :param epsilon: Privacy budget for differential privacy.
    :param delta_distortion: Maximum distortion for distortion-privacy.
    :param sensitivity: Sensitivity for differential privacy.
    :return: Adjusted sensing data with privacy-preserving mechanisms applied.
    """
    # Step 1: Apply differential privacy (Laplace mechanism)
    noisy_location = laplace_mechanism(location_data, epsilon, sensitivity)
    
    # Step 2: Apply distortion-privacy mechanism
    distorted_location = distort_location(noisy_location, delta_distortion)
    
    # Step 3: Adjust sensing data based on the obfuscated location
    adjusted_data = adjust_data_to_obfuscated_location(original_data, distorted_location)
    
    # Step 4: Solve the linear program for optimal location obfuscation
    optimized_location = optimize_location_obfuscation(location_data, epsilon, delta_distortion)
    
    # Step 5: Uncertainty-aware inference
    inferred_data, std_dev = uncertainty_aware_inference(distorted_location, original_data)
    
    # Return the results
    return adjusted_data, optimized_location, inferred_data, std_dev

# Example usage:
if __name__ == "__main__":
    # Example location data (coordinates) and sensing data (e.g., traffic counts, environmental data)
    location_data = np.array([10.1, 20.2, 30.3, 40.4, 50.5])  # Example coordinates
    original_data = np.array([100, 200, 150, 170, 180])  # Example sensor readings
    
    # Parameters for privacy mechanisms
    epsilon = 0.5  # Privacy budget (lower is more private)
    delta_distortion = 1.0  # Maximum allowed distortion
    sensitivity = 0.1  # Sensitivity for differential privacy
    
    # Run the privacy-enhanced MCS framework
    adjusted_data, optimized_location, inferred_data, std_dev = privacy_enhanced_mcs(
        location_data, original_data, epsilon, delta_distortion, sensitivity
    )
    
    # Display the results
    print("Adjusted Data:", adjusted_data)
    print("Optimized Obfuscated Location:", optimized_location)
    print("Inferred Data:", inferred_data)
    print("Standard Deviation of Inference:", std_dev)
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate MAE
def calculate_mae(actual_values, predicted_values):
    """
    Calculate Mean Absolute Error (MAE).
    :param actual_values: Actual values (e.g., true temperature readings).
    :param predicted_values: Predicted values based on obfuscation and budget allocation.
    :return: MAE value.
    """
    return np.mean(np.abs(actual_values - predicted_values))

# Function to simulate temperature sensing data for privacy-concerned and no-privacy participants
def simulate_data(num_privacy_concerned, num_no_privacy, actual_temperatures, cp, cn):
    """
    Simulate data for privacy-concerned and no-privacy participants.
    :param num_privacy_concerned: Number of privacy-concerned participants.
    :param num_no_privacy: Number of no-privacy participants.
    :param actual_temperatures: Actual temperature readings.
    :param cp: Cost per privacy-concerned participant.
    :param cn: Cost per no-privacy participant.
    :return: Predicted temperatures considering privacy and no-privacy participants.
    """
    # Apply noise for privacy-concerned participants
    noise_privacy_concerned = np.random.normal(0, 1, num_privacy_concerned)  # Simulated noise (privacy obfuscation)
    predicted_temperatures_privacy = actual_temperatures[:num_privacy_concerned] + noise_privacy_concerned
    
    # No noise for no-privacy participants
    predicted_temperatures_no_privacy = actual_temperatures[num_privacy_concerned:num_privacy_concerned+num_no_privacy]
    
    # Combine both sets of predicted temperatures
    predicted_temperatures = np.concatenate([predicted_temperatures_privacy, predicted_temperatures_no_privacy])
    
    return predicted_temperatures

# Main function to evaluate MAE for different budget partitions
def evaluate_mae_budget_partition(actual_temperatures, cp, cn, total_budget=12):
    """
    Evaluate the MAE for different budget partitions between privacy-concerned and no-privacy participants.
    :param actual_temperatures: Actual temperature readings.
    :param cp: Cost per privacy-concerned participant.
    :param cn: Cost per no-privacy participant.
    :param total_budget: Total budget to be divided between participants (default is 12).
    :return: MAE for each budget partition.
    """
    mae_values = []
    
    # Iterate over different partitions of the budget (number of privacy-concerned participants)
    for x in range(total_budget + 1):
        # Calculate the number of participants in each group
        num_privacy_concerned = x
        num_no_privacy = total_budget - x
        
        # Simulate the data for the given budget partition
        predicted_temperatures = simulate_data(num_privacy_concerned, num_no_privacy, actual_temperatures, cp, cn)
        
        # Calculate MAE for this partition
        mae = calculate_mae(actual_temperatures, predicted_temperatures)
        mae_values.append((x, mae))
    
    return mae_values

# Example usage:
if __name__ == "__main__":
    # Example actual temperature data (e.g., 12 temperature readings for 12 participants)
    actual_temperatures = np.array([20.5, 21.2, 19.8, 22.0, 21.5, 20.0, 21.7, 20.3, 21.0, 20.6, 19.5, 21.3])
    
    # Cost per participant (privacy-concerned and no-privacy)
    cp = 3  # Privacy-concerned cost
    cn = 1  # No-privacy cost
    
    # Total budget for participants
    total_budget = 12  # We have 12 participants
    
    # Evaluate MAE for different budget partitions
    mae_values = evaluate_mae_budget_partition(actual_temperatures, cp, cn, total_budget)
    
    # Extract x (privacy-concerned participants) and y (MAE values)
    x_values = [partition for partition, _ in mae_values]
    y_values = [mae for _, mae in mae_values]
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="MAE vs Budget Partition")
    plt.title("MAE vs Budget Partition for Privacy-Constrained Participants")
    plt.xlabel("Number of Privacy-Constrained Participants")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True)
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Function to calculate MAE
def calculate_mae(actual_values, predicted_values):
    return np.mean(np.abs(actual_values - predicted_values))

# Function to calculate MSE
def calculate_mse(actual_values, predicted_values):
    return np.mean((actual_values - predicted_values) ** 2)

# Function to calculate RMSE
def calculate_rmse(mse):
    return np.sqrt(mse)

# Function to simulate temperature sensing data for privacy-concerned and no-privacy participants
def simulate_data(num_privacy_concerned, num_no_privacy, actual_temperatures, cp, cn):
    noise_privacy_concerned = np.random.normal(0, 1, num_privacy_concerned)  # Simulated noise (privacy obfuscation)
    predicted_temperatures_privacy = actual_temperatures[:num_privacy_concerned] + noise_privacy_concerned
    predicted_temperatures_no_privacy = actual_temperatures[num_privacy_concerned:num_privacy_concerned+num_no_privacy]
    predicted_temperatures = np.concatenate([predicted_temperatures_privacy, predicted_temperatures_no_privacy])
    return predicted_temperatures

# Main function to evaluate MAE, MSE, RMSE, and R² for different budget partitions
def evaluate_metrics_budget_partition(actual_temperatures, cp, cn, total_budget=12):
    metrics = []
    
    for x in range(total_budget + 1):
        num_privacy_concerned = x
        num_no_privacy = total_budget - x
        predicted_temperatures = simulate_data(num_privacy_concerned, num_no_privacy, actual_temperatures, cp, cn)
        
        mae = calculate_mae(actual_temperatures, predicted_temperatures)
        mse = calculate_mse(actual_temperatures, predicted_temperatures)
        rmse = calculate_rmse(mse)
        r2 = r2_score(actual_temperatures, predicted_temperatures)
        
        metrics.append((x, mae, mse, rmse, r2))
    
    return metrics

# Example usage:
if __name__ == "__main__":
    actual_temperatures = np.array([20.5, 21.2, 19.8, 22.0, 21.5, 20.0, 21.7, 20.3, 21.0, 20.6, 19.5, 21.3])
    cp = 3  # Privacy-concerned cost
    cn = 1  # No-privacy cost
    total_budget = 12  # Total participants
    
    metrics = evaluate_metrics_budget_partition(actual_temperatures, cp, cn, total_budget)
    
    x_values = [partition for partition, _, _, _, _ in metrics]
    mae_values = [mae for _, mae, _, _, _ in metrics]
    mse_values = [mse for _, _, mse, _, _ in metrics]
    rmse_values = [rmse for _, _, _, rmse, _ in metrics]
    r2_values = [r2 for _, _, _, _, r2 in metrics]
    
    # Plot MAE, MSE, RMSE, R²
    plt.figure(figsize=(10, 8))
    
    # Plot MAE
    plt.subplot(2, 2, 1)
    plt.plot(x_values, mae_values, marker='o', color='b', label="MAE")
    plt.title("MAE vs Budget Partition")
    plt.xlabel("Number of Privacy-Constrained Participants")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True)
    
    # Plot MSE
    plt.subplot(2, 2, 2)
    plt.plot(x_values, mse_values, marker='o', color='r', label="MSE")
    plt.title("MSE vs Budget Partition")
    plt.xlabel("Number of Privacy-Constrained Participants")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(2, 2, 3)
    plt.plot(x_values, rmse_values, marker='o', color='g', label="RMSE")
    plt.title("RMSE vs Budget Partition")
    plt.xlabel("Number of Privacy-Constrained Participants")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.grid(True)
    
    # Plot R²
    plt.subplot(2, 2, 4)
    plt.plot(x_values, r2_values, marker='o', color='purple', label="R²")
    plt.title("R² vs Budget Partition")
    plt.xlabel("Number of Privacy-Constrained Participants")
    plt.ylabel("R-squared (R²)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
