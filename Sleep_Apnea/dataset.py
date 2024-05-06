
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Define number of samples
n_samples = 2000

# Generate random values for features
oxygen_levels = np.random.randint(80, 100, n_samples)
pulse_rates = np.random.randint(60, 100, n_samples)
ages = np.random.randint(30, 60, n_samples)  # Age between 30 and 59 years
heart_rates = np.random.randint(50, 120, n_samples)
ecg_signals = np.random.normal(0, 1, n_samples)  # Simulated ECG signals

# Introduce correlations for sleep apnea probabilities
sleep_apnea_probs = (oxygen_levels - 80) / 20 + (120 - pulse_rates) / 60 + (ages - 40) / 40 + (heart_rates - 70) / 50
sleep_apnea_probs = np.clip(sleep_apnea_probs, 0, 1)
sleep_apnea_labels = np.where(np.random.rand(n_samples) < sleep_apnea_probs, 'Yes', 'No')

# Create DataFrame
data = {
    'OxygenLevel': oxygen_levels,
    'Pulse': pulse_rates,
    'Age': ages,
    'HeartRate': heart_rates,
    'ECG': ecg_signals,
    'SleepApnea': sleep_apnea_labels
}

df = pd.DataFrame(data)

# Save DataFrame to CSV file
df.to_csv('sleep_apnea_data.csv', index=False)

# Display sample data
print("Sample Data from CSV File:")
# print(df.head())
print(df)
