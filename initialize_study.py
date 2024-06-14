import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
    plot_intermediate_values
)
import plotly.io as pio
import time
import os

def get_or_create_study(study_name, storage, direction):
    try:
        # Try to load the study
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Study '{study_name}' loaded.")
    except KeyError:
        # If the study does not exist, create it
        study = optuna.create_study(study_name=study_name, storage=storage, direction=direction)
        print(f"Study '{study_name}' created.")
    return study

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

study_name = 'yolov8_study_2'
storage = 'mysql+pymysql://optuna_user:optuna_password@192.168.25.182/optuna_db' #TODO: Replace with your database connection, the user and password
direction = 'minimize'

study = get_or_create_study(study_name, storage, direction)

# Set the refresh interval (in seconds)
refresh_interval = 60

while True:
    # Plot optimization history
    fig_optimization_history = plot_optimization_history(study)
    pio.write_html(fig_optimization_history, file='plots/optimization_history.html', auto_open=False)
    print("optimization_history")

    # Plot hyperparameter importance
    try:
        fig_param_importances = plot_param_importances(study)
        pio.write_html(fig_param_importances, file='plots/param_importances.html', auto_open=False)
        print("param_importances")
    except ValueError:
        print("Value error - might be only one data point - continue")

    # Plot parallel coordinate
    fig_parallel_coordinate = plot_parallel_coordinate(study)
    pio.write_html(fig_parallel_coordinate, file='plots/parallel_coordinate.html', auto_open=False)
    print("parallel_coordinate")

    # Plot contour plot
    fig_contour = plot_contour(study)
    pio.write_html(fig_contour, file='plots/contour.html', auto_open=False)
    print("contour")
    
    # Plot slice plot
    fig_slice = plot_slice(study)
    pio.write_html(fig_slice, file='plots/slice.html', auto_open=False)
    print("slice")

    # Plot intermediate values
    #fig_intermediate_values = plot_intermediate_values(study)
    #pio.write_html(fig_intermediate_values, file='plots/intermediate_values.html', auto_open=False)

    print("Visualizations updated. Refresh your browser to see the latest results.")
    
    # Wait for the specified interval before refreshing
    time.sleep(refresh_interval)

