from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any
from numpy.typing import NDArray
import numpy as np
import torch

class Imputer(ABC):
    ''' ... '''
    def __init__(self,
        modalities: dict[str, dict[str, Any]],
        is_embedding: dict[str, bool] | None = None
    ) -> None:
        ''' ... '''
        self.modalities = modalities
        self.is_embedding = is_embedding
    
    @abstractmethod
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
    ) -> dict[str, int | NDArray[np.float32]]:
        ''' ... '''
        pass

    @staticmethod
    def _keyerror_hint(func):
        ''' Print hint for resolving KeyError. '''
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError as err:
                raise ValueError('Format the data using Formatter module.') from err
        return wrapper


class ConstantImputer(Imputer):
    ''' ... '''
    @Imputer._keyerror_hint
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
    ) -> dict[str, int | NDArray[np.float32]]:
        ''' ... '''
        new = dict()
        for k, info in self.modalities.items():
            if smp[k] is not None:
                new[k] = smp[k]
            else:
                if self.is_embedding is not None and k in self.is_embedding and self.is_embedding[k]:
                    new[k] = np.zeros(256, dtype=np.float32)
                else:
                    if info['type'] == 'categorical':
                        new[k] = 0
                    elif info['type'] == 'numerical' or info['type'] == 'imaging':
                        new[k] = np.zeros(tuple(info['shape']), dtype=np.float32)
                    else:
                        raise ValueError
        return new
                

class FrequencyImputer(Imputer):
    ''' ... '''
    @Imputer._keyerror_hint
    def __init__(self, 
        modalities: dict[str, dict[str, Any]],
        dat: list[dict[str, int | NDArray[np.float32] | None]],
    ) -> None:
        ''' ... '''
        super().__init__(modalities)

        # List[Dict] to Dict[List]
        self.pool = {k: [smp[k] for smp in dat] for k in modalities}
        
        # remove None
        self.pool = {k: [v for v in self.pool[k] if v is not None] for k in self.pool}
        
        
    @Imputer._keyerror_hint
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
    ) -> dict[str, int | NDArray[np.float32]]:
        ''' ... '''
        new = dict()
        for k, info in self.modalities.items():
            if smp[k] is not None:
                new[k] = smp[k]
            else:
                # print(k)
                if info['type'] == 'categorical':
                    new[k] = 0
                else:
                    if info['type'] == 'numerical':
                        rnd_idx = np.random.randint(0, len(self.pool[k]))
                        new[k] = np.array(self.pool[k][rnd_idx])
                        # print(type(new[k]))
                    elif info['type'] == 'imaging':
                        new[k] = np.zeros(tuple(info['shape']), dtype=np.float32)
                        # print(new[k].shape)
                    else:
                        ic(info['shape'])
                        raise ValueError
        return new