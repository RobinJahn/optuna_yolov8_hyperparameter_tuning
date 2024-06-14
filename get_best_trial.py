import optuna

# Load the study
study = optuna.load_study(
    study_name='yolov8_study',  # Replace with your study name
    storage='mysql+pymysql://optuna_user:optuna_password@192.168.25.182/optuna_db' #TODO: Replace with your database connection, the user and password
)

# Get the best trial
best_trial = study.best_trial

# Print the best parameters
print("Best parameters found:")
for key, value in best_trial.params.items():
    print(f"        '{key}': {value},")

print("\nContinious list:")
for key, value in best_trial.params.items():
    print(f"{key}={value} ", end="")
print()

# Print the best value (objective value)
print(f"\nBest value (objective): {best_trial.value}")
