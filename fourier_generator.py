import numpy as np
import matplotlib.pyplot as plt
import os

class FourierSeriesGenerator:

    def __init__(self, total_time=400, time_step=0.002):
        self.default_total_time = total_time
        self.default_time_step = time_step

    @staticmethod
    def _normalize(x):
        return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    @staticmethod
    def _rescale(x, min_value, max_value):
        return x * (max_value - min_value) + min_value

    def fourier_series(self, x, params, rescale_mag=600, rescale_amplitude=50):
        x = self._normalize(x)

        n, freq, amplitude, phase, trend, seasonality, = params
        n = int(self._rescale(n, 0, 10))
        freq = self._rescale(freq, 0, 10)
        amplitude = self._rescale(amplitude, 0, 10)
        phase = self._rescale(phase, 0, 10000)
        trend = self._rescale(trend, -500, 500)
        seasonality = self._rescale(seasonality, 0, 200)

        sum = np.zeros_like(x)
        for i in range(1, n + 1, 2):
            term = (1 / i) * np.sin(2 * np.pi * freq * i * x + phase)
            sum += term

        y = amplitude * (2 / np.pi) * sum
        if np.sum(y) == 0:
            return np.zeros_like(x) + 600
        else:
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            y = (y * rescale_amplitude) + rescale_mag

        y += trend * x
        y += seasonality * np.sin(2 * np.pi * x)
        return y

    def plot_and_save(self, params, base_path, iteration, total_time=None, time_step=None):
        if total_time is None:
            total_time = self.default_total_time
        if time_step is None:
            time_step = self.default_time_step

        folder_name = f"Iteration_{iteration}"
        save_directory = os.path.join(base_path, folder_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  # Create directory if it doesn't exist

        x = np.linspace(0, total_time, int(total_time / time_step))
        y = self.fourier_series(x, params)

        plt.plot(x, y, label=f'Curve')
        plt.xlabel("Time")
        plt.ylabel("Laser Power")
        plt.legend()
        image_path = os.path.join(save_directory, "plot.png")
        plt.savefig(image_path)
        plt.show()

        output_string = "laser_power,time_elapsed\n"
        for i in range(len(x)):
            output_string += f"{y[i]:.15f},{x[i]:.2f}\n"
        csv_path = os.path.join(save_directory, "data.csv")
        with open(csv_path, "w") as f:
            f.write(output_string)