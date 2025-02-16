import numpy as np
from abc import ABC, abstractmethod

import torch


class PhysicsProcess(ABC):

    def __init__(self, process_signature, store_params, max_noise):
        self.params = {}
        self.noise_factor = {}
        self.signature = process_signature
        self.store_params = store_params
        self.max_noise = max_noise

    def get_data(self, label, delta_t, N):
        if self.store_params:
            if label not in self.params.keys():
                self.params[label] = self.get_random_params()
                self.noise_factor[label] = self.get_random_noise_factor()
            p = self.params[label]
            n = self.noise_factor[label]
        else:
            p = self.get_random_params()
            n = self.get_random_noise_factor()
        dynamics = torch.Tensor(self.generate_data(delta_t, N, **p))
        noise = torch.normal(0, torch.abs(dynamics) * n)
        return dynamics + noise

    def get_params(self, label):
        result = self.params.get(label)
        if result is None:
            return result
        result["noise"] = self.noise_factor.get(label)
        result["process"] = self.signature
        return result

    def get_random_noise_factor(self):
        return np.random.uniform(0, self.max_noise, 1)

    @abstractmethod
    def get_random_params(self):
        pass

    @abstractmethod
    def generate_data(self, delta_t, N, **kwargs):
        pass


# 1. Harmonic Oscillation
class HarmonicOscillation(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Harmonic Oscillation", *args)

    def get_random_params(self):
        return {'amplitude': 1, 'omega': 2 * np.pi, 'phase': 0}

    def generate_data(self, delta_t, N, amplitude, omega, phase):
        t = np.arange(0, delta_t * N, delta_t)
        return amplitude * np.cos(omega * t + phase)


# 2. Pendulum Motion
class PendulumMotion(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Pendulum Motion", *args)

    def get_random_params(self):
        return {'length': 1, 'g': 9.81, 'theta0': 0.1}

    def generate_data(self, delta_t, N, length, g, theta0):
        omega = np.sqrt(g / length)
        t = np.arange(0, delta_t * N, delta_t)
        return theta0 * np.cos(omega * t)


# 3. Electrical Circuit Oscillation
class CircuitOscillation(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Electrical Circuit Oscillation", *args)

    def get_random_params(self):
        return {'L': 1, 'C': 1, 'V0': 1}

    def generate_data(self, delta_t, N, L, C, V0):
        omega = 1 / np.sqrt(L * C)
        t = np.arange(0, delta_t * N, delta_t)
        return V0 * np.cos(omega * t)


# 4. Heat Conduction
class HeatConduction(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Heat Conduction", *args)

    def get_random_params(self):
        return {'initial_temp': 100, 'alpha': 0.01, 'x': 0.5}

    def generate_data(self, delta_t, N, initial_temp, alpha, x):
        t = np.arange(0, delta_t * N, delta_t)
        return initial_temp * np.exp(-alpha * (np.pi ** 2) * t) * np.cos(np.pi * x)


# 5. Radioactive Decay
class RadioactiveDecay(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Radioactive Decay", *args)

    def get_random_params(self):
        return {'N0': 100, 'decay_const': 0.1}

    def generate_data(self, delta_t, N, N0, decay_const):
        t = np.arange(0, delta_t * N, delta_t)
        return N0 * np.exp(-decay_const * t)


# 6. Projectile Motion
class ProjectileMotion(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Projectile Motion", *args)

    def get_random_params(self):
        return {'v0': 10, 'g': 9.81}

    def generate_data(self, delta_t, N, v0, g):
        t = np.arange(0, delta_t * N, delta_t)
        return v0 * t - 0.5 * g * t ** 2


# 7. Planetary Orbital Motion
class OrbitalMotion(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Planetary Orbital Motion", *args)

    def get_random_params(self):
        return {'semi_major_axis': 1, 'eccentricity': 0.1, 'period': 1}

    def generate_data(self, delta_t, N, semi_major_axis, eccentricity, period):
        t = np.arange(0, delta_t * N, delta_t)
        omega = 2 * np.pi / period
        return semi_major_axis * (1 - eccentricity * np.cos(omega * t))


# 8. Free Fall in a Viscous Medium
class FreeFall(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Free Fall in a Viscous Medium", *args)

    def get_random_params(self):
        return {'m': 1, 'g': 9.81, 'b': 0.1}

    def generate_data(self, delta_t, N, m, g, b):
        t = np.arange(0, delta_t * N, delta_t)
        terminal_velocity = m * g / b
        return terminal_velocity * (1 - np.exp(-b * t / m))


# 9. Wave Propagation
class WavePropagation(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Wave Propagation", *args)

    def get_random_params(self):
        return {'amplitude': 1, 'frequency': 1}

    def generate_data(self, delta_t, N, amplitude, frequency):
        t = np.arange(0, delta_t * N, delta_t)
        return amplitude * np.sin(2 * np.pi * frequency * t)


# 10. Electrical Resistance Heating
class ResistanceHeating(PhysicsProcess):
    def __init__(self, *args):
        super().__init__("Electrical Resistance Heating", *args)

    def get_random_params(self):
        return {'power': 10, 'mass': 1, 'specific_heat': 1, 'T0': 20}

    def generate_data(self, delta_t, N, power, mass, specific_heat, T0):
        t = np.arange(0, delta_t * N, delta_t)
        return T0 + (power / (mass * specific_heat)) * t


def physics_process_factory(store_params, max_noise):
    return [
        HarmonicOscillation(store_params, max_noise),
        PendulumMotion(store_params, max_noise),
        CircuitOscillation(store_params, max_noise),
        HeatConduction(store_params, max_noise),
        RadioactiveDecay(store_params, max_noise),
        ProjectileMotion(store_params, max_noise),
        OrbitalMotion(store_params, max_noise),
        FreeFall(store_params, max_noise),
        WavePropagation(store_params, max_noise),
        ResistanceHeating(store_params, max_noise),
    ]
