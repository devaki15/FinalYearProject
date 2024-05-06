
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

def detect_sleep_apnea():
    # Load user inputs
    oxygen_level = float(entry_oxygen.get())
    pulse = float(entry_pulse.get())
    age = int(entry_age.get())
    heart_rate = int(entry_heart_rate.get())
    ecg_signal = float(entry_ecg.get())


    # Preprocess input data
    input_data = pd.DataFrame({
        'OxygenLevel': [oxygen_level],
        'Pulse': [pulse],
        'Age': [age],
        'HeartRate': [heart_rate],
        'ECG': [ecg_signal]
    })

    # Load the trained model
    model = joblib.load('sleep_apnea_model.pkl')

    # Use the model to predict probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Determine sleep apnea severity based on probabilities
    if probabilities[1] >= 0.8:
        severity = "Severe"
        suggestions = "Consult a sleep specialist immediately. Consider using a CPAP machine."
    elif probabilities[1] >= 0.5:
        severity = "Moderate"
        suggestions = "Schedule an appointment with a sleep doctor. Avoid alcohol and tobacco before bedtime."
    elif probabilities[1] >= 0.3:
        severity = "Mild"
        suggestions = "Make lifestyle changes such as losing weight and avoiding sleeping on your back."
    else:
        severity = "None"
        suggestions = "No significant sleep apnea detected. Maintain a healthy lifestyle."

    # Show prediction result with severity and suggestions
    messagebox.showinfo("Result", f"Sleep Apnea: {severity} ({probabilities[1]*100:.2f}% probability)\n\nSuggestions:\n{suggestions}")

# Create GUI window
window = tk.Tk()
window.title("Sleep Apnea Detector")
window.geometry("400x300")  # Set initial window size (width x height)

# Add input fields
tk.Label(window, text="Oxygen Level:").grid(row=0)
tk.Label(window, text="Pulse:").grid(row=1)
tk.Label(window, text="Age:").grid(row=2)
tk.Label(window, text="Heart Rate:").grid(row=3)
tk.Label(window, text="ECG  Signal:").grid(row=4)

entry_oxygen = tk.Entry(window)
entry_pulse = tk.Entry(window)
entry_age = tk.Entry(window)
entry_heart_rate = tk.Entry(window)
entry_ecg = tk.Entry(window)
entry_oxygen.grid(row=0, column=1, padx=10, pady=5)
entry_pulse.grid(row=1, column=1, padx=10, pady=5)
entry_age.grid(row=2, column=1, padx=10, pady=5)
entry_heart_rate.grid(row=3, column=1, padx=10, pady=5)
entry_ecg.grid(row=4, column=1, padx=10, pady=5)

# Add empty labels for spacing
tk.Label(window, text="").grid(row=4)  # Spacer row

# Add button to detect sleep apnea
detect_button = tk.Button(window, text="Detect Sleep Apnea", command=detect_sleep_apnea)
detect_button.grid(row=5, columnspan=2, pady=10)
window.mainloop()
