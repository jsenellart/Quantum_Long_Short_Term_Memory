import numpy as np
import math
from scipy.integrate import odeint
from scipy.special import jv
from typing import Tuple

from .base import DataGenerator

class SinGenerator(DataGenerator):
    def __init__(self, frequency=1.0, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
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
    def __init__(self, frequency=1.0, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
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
    def __init__(self, amplitude=1.0, x_scale=1.0, x_max=20, n_points=240, noise_std=0.0):
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