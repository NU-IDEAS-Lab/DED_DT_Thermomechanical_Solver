import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd
from tqdm import tqdm
from gamma_model_simulator import GammaModelSimulator
import os

class TemperatureAnalyzer:

    def __init__(self, input_data_dir, sim_dir_name, laser_file):
        self.INPUT_DATA_DIR = input_data_dir
        self.SIM_DIR_NAME = sim_dir_name
        self.LASER_FILE = laser_file

    def calculate_time(self, df, min_temp, max_temp, selected_nodes, collection_rate=0.02, plot_graph=False):  # Notice 'self' added as the first argument
        total_time_list = []
        
        for column in tqdm(df.columns, desc="Processing nodes"):
            time_axis = np.arange(0, df[column].size * collection_rate, collection_rate)
            
            # Find indices where temperature is within the desired range
            in_range_indices = np.where((df[column] >= min_temp) & (df[column] <= max_temp))[0]
            
            # Check if there are any in-range values
            if len(in_range_indices) == 0:
                total_time_list.append(0)
                continue

            # Calculate time between first and last in-range value for this column
            time_diff = (in_range_indices[-1] - in_range_indices[0]) * collection_rate
            total_time_list.append(time_diff)
            
            # If plotting is enabled and this column is one of the selected nodes, then plot
            if plot_graph and str(column) in selected_nodes:
                plt.figure(figsize=(10,5))
                plt.plot(time_axis, df[column], label=f"Node {column}")
                if len(in_range_indices) > 0:
                    plt.fill_between(time_axis, 
                                    min_temp, 
                                    max_temp, 
                                    where=((df[column] >= min_temp) & (df[column] <= max_temp)),
                                    color='gray', alpha=0.5, label=f"Temp between {min_temp} and {max_temp}")
                
                # Adding horizontal lines for min and max temperature
                plt.axhline(min_temp, color='red', linestyle='--', label=f"Min Temp {min_temp}")
                plt.axhline(max_temp, color='blue', linestyle='--', label=f"Max Temp {max_temp}")

                plt.xlabel("Time (seconds)")
                plt.ylabel("Temperature")
                plt.title(f"Temperature vs Time for Node {column}")
                plt.legend()
                plt.grid(True)

                # Save the figure to the same directory as the zarr file, with a specific filename for the node
                #figure_path = os.path.join(os.path.dirname(zarr_location), f"Node_{column}_Temperature_vs_Time.png")
                #plt.savefig(figure_path)  # Save the figure first
                plt.show()  # Then show the figure
            
        return np.mean(total_time_list)         

    def run_simulation_and_analyze(self, zarr_location, min_temp, max_temp, selected_nodes_list, collection_rate=0.02, plot_graph=False):
        # Create an instance of the simulator
        simulator = GammaModelSimulator(
            input_data_dir=self.INPUT_DATA_DIR,
            sim_dir_name=self.SIM_DIR_NAME,
            laser_file=self.LASER_FILE)

        # Set up the simulation
        simulator.setup_simulation()

        # Run the simulation
        simulator.run_simulation()

        # Open the zarr file
        zarr_array = zarr.open(zarr_location, mode='r')

        # Convert the zarr array into a pandas DataFrame
        df = pd.DataFrame(zarr_array[:])

        return self.calculate_time(df, min_temp, max_temp, selected_nodes_list, collection_rate, plot_graph)

