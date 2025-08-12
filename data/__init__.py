from typing import Dict, Type, Any
from .base import DataGenerator
from .generators import SinGenerator, CosGenerator, LinearGenerator, ExponentialGenerator, DampedSHMGenerator,\
                        BesselJ2Generator, AirlinePassengersGenerator,\
                        PopulationInversionGenerator, PopulationInversionCollapseRevivalGenerator

class DataFactory:
    _generators: Dict[str, Type[DataGenerator]] = {
        'damped_shm': DampedSHMGenerator,
        'sin': SinGenerator,
        'cos': CosGenerator,
        'linear': LinearGenerator,
        'exp': ExponentialGenerator,
        'exponential': ExponentialGenerator,
        'bessel_j2': BesselJ2Generator,
        'airline_passengers': AirlinePassengersGenerator,
        'population_inversion': PopulationInversionGenerator,
        'population_inversion_collapse_revival': PopulationInversionCollapseRevivalGenerator
    }

    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[DataGenerator]):
        cls._generators[name] = generator_class
    
    @classmethod
    def get(cls, name: str, **kwargs) -> DataGenerator:
        if name not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Unknown generator '{name}'. Available: {available}")
        
        generator_class = cls._generators[name]
        return generator_class(**kwargs)
    
    @classmethod
    def list_generators(cls) -> list:
        return list(cls._generators.keys())

data = DataFactory()
