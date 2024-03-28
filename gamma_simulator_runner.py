from gamma_model_simulator import GammaModelSimulator

class GammaSimulatorRunner:
    def __init__(self, input_data_dir, sim_dir_name, laser_file):
        self.simulator = GammaModelSimulator(
            input_data_dir=input_data_dir,
            sim_dir_name=sim_dir_name,
            laser_file=laser_file
        )

    def run(self):
        # Set up the simulation
        self.simulator.setup_simulation()

        # Execute the simulation
        self.simulator.run_simulation()

if __name__ == "__main__":
    # Define your parameters
    INPUT_DATA_DIR = "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data"
    SIM_DIR_NAME = "thin_wall"
    LASER_FILE = "/home/vnk3019/ded_dt_thermomechanical_solver/examples/data/laser_inputs/thin_wall/LP_2"

    runner = GammaSimulatorRunner(INPUT_DATA_DIR, SIM_DIR_NAME, LASER_FILE)
    runner.run()
