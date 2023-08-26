import numpy as np
import pandas as pd
import zarr
from botorch.fit import fit_gpytorch_model
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings
import shutil

warnings.filterwarnings("ignore")


# This function deletes a directory and all its contents
def delete_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted: {dir_path}")
    else:
        print(f"The directory {dir_path} does not exist")

# Define the neural network parameters in the following function:
def objective(params, iteration_number, min_temp=893, max_temp=993):
    from fourier_generator import FourierSeriesGenerator
    from temperature_analyzer import TemperatureAnalyzer


    # Generate and save the Fourier series
    generator = FourierSeriesGenerator()
    base_path = "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data/laser_inputs/thin_wall"
    generator.plot_and_save(params, base_path, iteration_number, total_time=300, time_step=0.002)

    # Paths for the simulator
    INPUT_DATA_DIR = "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data"
    SIM_DIR_NAME = "thin_wall"

    # Modify LASER_FILE to reflect the correct iteration and CSV filename
    LASER_FILE = os.path.join(base_path, f"Iteration_{iteration_number}", "data")

    # Modify ZARR_LOCATION to reflect the correct iteration
    ZARR_LOCATION = os.path.join(base_path, f"Iteration_{iteration_number}", "data.zarr", "ff_dt_temperature")
    
    selected_nodes_list = ["45003"]  # As an example

    analyzer = TemperatureAnalyzer(
        INPUT_DATA_DIR,
        SIM_DIR_NAME,
        LASER_FILE
    )

    # Call function
    avg_time = analyzer.run_simulation_and_analyze(ZARR_LOCATION, min_temp, max_temp, selected_nodes_list, plot_graph=True)
    return torch.tensor(avg_time)  # Now returns a 1D tensor



# Bayesian Optimization functions
# -----------------------------------------------------------
def initialize_model():
    train_X = pd.read_excel("optimized_params.xlsx")
    train_X_np = train_X[["n", "freq", "amplitude", "phase", "trend", "seasonality"]].values
    train_X_torch = torch.tensor(train_X_np, dtype=torch.float32)
    train_Y = pd.read_excel("avg_heat_treatment_times.xlsx")
    train_Y_np = train_Y[["Average Heat Treatment Time"]].values
    train_Y_torch = torch.tensor(train_Y_np, dtype=torch.float32)
    gp = SingleTaskGP(train_X_torch.float(), train_Y_torch.float())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    return gp

# Function to compute adaptive beta
def adaptive_beta(i, uncertainties, k=2.0, uncertainty_factor=0.5):
    # Use inverse square root decay for beta
    beta_base = k / np.sqrt(i + 1)

    # Adjust beta based on recent uncertainty
    if uncertainties:
        recent_uncertainty = np.mean(uncertainties[-5:])  # average over last 5 steps
        beta_adjusted = beta_base * (1 + uncertainty_factor * recent_uncertainty)
    else:
        beta_adjusted = beta_base

    return beta_adjusted


def optimize(bounds, n_steps=50):
    gp = initialize_model()
    best_value = -float('inf')
    best_params = None

    param_history = []
    value_history = []
    uncertainty_history = []

    # Create an empty dataframe with the required columns
    df = pd.DataFrame(columns=['Parameters', 'Objective Value', 'Uncertainty'])

    for i in tqdm(range(n_steps)):
        if i > 0:
            delete_dir_path = os.path.join(
                "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data/laser_inputs/thin_wall", 
                f"Iteration_{i-1}","data"
            )
            delete_directory(delete_dir_path)
        if i > 0:
            delete_dir_path = os.path.join(
                "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data/laser_inputs/thin_wall", 
                f"Iteration_{i-1}","data.zarr"
            )
            delete_directory(delete_dir_path)
        # Dynamic beta based on iteration number and uncertainty
        beta = adaptive_beta(i, uncertainty_history)
        UCB = UpperConfidenceBound(gp, beta=beta)        
        candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        candidate_numpy = candidate.detach().numpy().flatten()
        new_Y = objective(candidate_numpy, iteration_number=i).unsqueeze(0).unsqueeze(-1)

        variance = gp.posterior(candidate).variance
        uncertainty_history.append(variance.item())

        print(new_Y)
        print(candidate_numpy)
        

        if new_Y.item() > best_value:
            best_value = new_Y.item()
            best_params = candidate_numpy

        param_history.append(candidate_numpy)
        value_history.append(new_Y.item())

        # Update the Gaussian Process model
        gp = SingleTaskGP(
            torch.cat([gp.train_inputs[0], torch.tensor(candidate_numpy.astype(np.float32)).unsqueeze(0)]),
            torch.cat([gp.train_targets.unsqueeze(-1).float(), new_Y.float()], dim=0)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Save the current iteration results to the dataframe
        df = df.append({
            'Current Best Parameters': best_params,
            'Current Best Value': best_value, 
            'Parameters': candidate_numpy.tolist(),
            'Objective Value': new_Y.item(),
            'Uncertainty': variance.item()
        }, ignore_index=True)

        # Save the dataframe to an Excel file
        df.to_csv('bayesian_optimization_results.csv', index=False)

    return gp, best_params, best_value, param_history, value_history, uncertainty_history
# -----------------------------------------------------------

if __name__ == "__main__":
    input_size = 6  # Assuming 6 parameters
    bounds = torch.tensor([[0]*input_size, [1]*input_size], dtype=torch.float32)
    optimized_model, best_params, best_value, param_history, value_history, uncertainty_history = optimize(bounds)

    print(f'Best parameters: {best_params}, Best value: {best_value}')
    print('Uncertainty at each step:', uncertainty_history)
