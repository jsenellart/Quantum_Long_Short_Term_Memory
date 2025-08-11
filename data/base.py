import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class DataGenerator(ABC):
    def __init__(self, name: str):
        self.name = name
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self._raw_data = None
        self._normalized_data = None
        self._time_steps = None
        
    @abstractmethod
    def generate_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    def normalize_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        if data is None:
            if self._raw_data is None:
                self._time_steps, self._raw_data = self.generate_raw_data()
            data = self._raw_data
            
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self._normalized_data = self.scaler.fit_transform(data)
        return self._normalized_data
    
    def transform_data_single_predict(self, data: Optional[np.ndarray] = None, seq_length: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        if data is None:
            if self._normalized_data is None:
                data = self.normalize_data()
            else:
                data = self._normalized_data
        
        x = []
        y = []
        
        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length)]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)
            
        x_var = Variable(torch.from_numpy(np.array(x).reshape(-1, seq_length)).float())
        y_var = Variable(torch.from_numpy(np.array(y)).float())
        
        return x_var, y_var
    
    def plot(self, data: Optional[np.ndarray] = None, time_steps: Optional[np.ndarray] = None):
        if data is None:
            if self._normalized_data is None:
                self.normalize_data()
            data = self._normalized_data
        
        if time_steps is None:
            if self._time_steps is None:
                self._time_steps, _ = self.generate_raw_data()
            time_steps = self._time_steps
        
        fig = plt.figure(1, figsize=(9, 8))
        ax1 = fig.add_subplot()
        ax1.plot(time_steps, data)
        ax1.axhline(color="grey", ls="--", zorder=-1)
        ax1.set_ylim(-1, 1)
        ax1.text(0.5, 0.95, self.name, ha='center', va='top', transform=ax1.transAxes)
        plt.show()
    
    def get_data(self, seq_len: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform_data_single_predict(seq_length=seq_len)