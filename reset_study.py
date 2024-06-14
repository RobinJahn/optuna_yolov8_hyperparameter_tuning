import optuna
from optuna.exceptions import KeyError

def reset_study(study_name, storage, direction):
    try:
        # Try to delete the study if it exists
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Study '{study_name}' deleted.")
    except KeyError:
        print(f"Study '{study_name}' does not exist, so it cannot be deleted.")
    
    # Create a new study
    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction)
    print(f"Study '{study_name}' created.")
    return study

# Example usage
study_name = 'yolov5_study'
storage = 'mysql+pymysql://optuna_user:optuna_password192.168.25.182/optuna_db' #TODO: Replace with your database connection, the user and password
direction = 'minimize'

print("Resetting study...")
study = reset_study(study_name, storage, direction)
print("Study reset and ready for optimization.")

