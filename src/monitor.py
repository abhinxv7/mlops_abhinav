import pandas as pd
import numpy as np
import subprocess
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Load your original training data (Reference Data)
reference_data = pd.read_csv('data/churn.csv')
reference_data['TotalCharges'] = pd.to_numeric(reference_data['TotalCharges'], errors='coerce').fillna(0)

# 2. Create fake "future" data (Current Data)
current_data = reference_data.copy()

# 3. Inject Synthetic Drift (The Trap!)
drift_size = int(len(current_data) * 0.20)
current_data.loc[0:drift_size, 'MonthlyCharges'] = current_data.loc[0:drift_size, 'MonthlyCharges'] * 2
print("Trap set! 20% of the data has been corrupted.")

# 4. Set up and run the Watchdog (Evidently AI)
print("Running the Watchdog...")
drift_report = Report(metrics=[DataDriftPreset(drift_share=0.01)])
drift_report.run(reference_data=reference_data, current_data=current_data)

# Save the visual dashboard
drift_report.save_html("drift_dashboard.html")
print("Visual dashboard saved to drift_dashboard.html.")

# 5. AUTOMATED RETRAINING LOGIC (The New Feature)
# Convert the report into a dictionary we can read programmatically
report_dict = drift_report.as_dict()

# Extract the specific True/False value for overall dataset drift
drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]

if drift_detected:
    print("\n🚨 ALERT: Significant Data Drift Detected!")
    print("Initiating automated model retraining pipeline...\n")
    
    # This command tells your computer to run the train.py script automatically
    subprocess.run(["python", "src/train.py"])
    
    print("\n✅ Automated retraining complete. The new brain is ready for deployment.")
else:
    print("\n✅ No drift detected. Model is healthy and does not need retraining.")