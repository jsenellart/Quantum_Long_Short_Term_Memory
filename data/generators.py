import pandas as pd

import numpy as np
import math
import os
from scipy.integrate import odeint
from scipy.special import jv
from typing import Tuple

from .base import DataGenerator

class SinGenerator(DataGenerator):
    def __init__(self, frequency=0.2, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__("Sine Wave")
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.t_max = t_max
        self.n_points = n_points
        self.noise_std = noise_std
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        data = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return t, data

class CosGenerator(DataGenerator):
    def __init__(self, frequency=0.2, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__("Cosine Wave")
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.t_max = t_max
        self.n_points = n_points
        self.noise_std = noise_std
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        data = self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return t, data

class LinearGenerator(DataGenerator):
    def __init__(self, slope=1.0, intercept=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__("Linear")
        self.slope = slope
        self.intercept = intercept
        self.t_max = t_max
        self.n_points = n_points
        self.noise_std = noise_std
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        data = self.slope * t + self.intercept
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return t, data

class ExponentialGenerator(DataGenerator):
    def __init__(self, growth_rate=0.1, initial_value=1.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__("Exponential")
        self.growth_rate = growth_rate
        self.initial_value = initial_value
        self.t_max = t_max
        self.n_points = n_points
        self.noise_std = noise_std
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        data = self.initial_value * np.exp(self.growth_rate * t)
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return t, data

class DampedSHMGenerator(DataGenerator):
    def __init__(self, b=0.15, g=9.81, l=1, m=1, theta_0=None, t_max=20, n_points=240):
        super().__init__("Damped SHM")
        self.b = b
        self.g = g
        self.l = l
        self.m = m
        self.theta_0 = theta_0 if theta_0 is not None else [0, 3]
        self.t_max = t_max
        self.n_points = n_points
    
    def _system(self, theta, t, b, g, l, m):
        theta1 = theta[0]
        theta2 = theta[1]
        dtheta1_dt = theta2
        dtheta2_dt = -(b/m)*theta2 - g*math.sin(theta1)
        dtheta_dt = [dtheta1_dt, dtheta2_dt]
        return dtheta_dt
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        theta = odeint(self._system, self.theta_0, t, args=(self.b, self.g, self.l, self.m))
        return t, theta[:, 1]

class BesselJ2Generator(DataGenerator):
    def __init__(self, amplitude=1.0, x_scale=5.0, x_max=20, n_points=240, noise_std=0.0):
        super().__init__("Bessel J_2")
        self.amplitude = amplitude
        self.x_scale = x_scale
        self.x_max = x_max
        self.n_points = n_points
        self.noise_std = noise_std
    
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.1, self.x_max, self.n_points)  # Start from 0.1 to avoid singularity at x=0
        scaled_x = self.x_scale * x
        data = self.amplitude * jv(2, scaled_x)  # Bessel function of the first kind, order 2
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return x, data

class AirlinePassengersGenerator(DataGenerator):
    def __init__(self, csv_path=None, url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", n_points=None):
        super().__init__("Airline Passengers")
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "airline-passengers.csv")
        self.csv_path = csv_path
        self.url = url
        self.n_points = n_points
        self._df = None
        self._load_data()

    def _load_data(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            print(f"Downloading {self.csv_path}...")
            df = pd.read_csv(self.url, index_col='Month', parse_dates=True)
            df.to_csv(self.csv_path)
        self._df = pd.read_csv(self.csv_path, index_col='Month', parse_dates=True)
        if self.n_points is not None:
            self._df = self._df.iloc[:self.n_points]

    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(len(self._df))
        data = self._df['Passengers'].values.astype(float)
        return t, data

class PopulationInversionGenerator(DataGenerator):
    """
    Generator for the population inversion of a two-level system (Jaynes-Cummings model),
    as in https://arxiv.org/pdf/2009.01783 (see Eq. 1 and related text).
    W(t) = cos(omega * t)
    """
    def __init__(self, omega=1.0, amplitude=1.0, t_max=1000, n_points=240, noise_std=0.0):
        super().__init__("Population Inversion (Jaynes-Cummings)")
        self.omega = omega
        self.amplitude = amplitude
        self.t_max = t_max
        self.n_points = n_points
        self.noise_std = noise_std

    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        data = self.amplitude * np.cos(self.omega * t)
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, len(data))
        return t, data

from scipy.stats import poisson

class PopulationInversionCollapseRevivalGenerator(DataGenerator):
    """
    Generator for Jaynes-Cummings population inversion with collapse and revival (Eq. 15 in arXiv:2009.01783).
    W(t) = sum_n P_n * cos(2 * g * t * sqrt(n+1)),
    where P_n is Poissonian (mean_n), g is the coupling.
    """
    def __init__(self, mean_n=40, g=1.0, t_max=200, n_points=2500, n_max=100, noise_std=0.0):
        super().__init__("Population Inversion (Collapse & Revival)")
        self.mean_n = mean_n
        self.g = g
        self.t_max = t_max
        self.n_points = n_points
        self.n_max = n_max if n_max is not None else int(mean_n + 8 * np.sqrt(mean_n))  # cover most of the Poisson weight
        self.noise_std = noise_std

    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.t_max, self.n_points)
        n_vals = np.arange(0, self.n_max+1)
        P_n = poisson.pmf(n_vals, self.mean_n)
        # Outer product: shape (len(t), len(n_vals))
        cos_terms = np.cos(2 * self.g * np.outer(t, np.sqrt(n_vals+1)))
        # Weighted sum over n for each t
        W_t = np.dot(cos_terms, P_n)
        if self.noise_std > 0:
            W_t += np.random.normal(0, self.noise_std, len(W_t))
        return t, W_t
