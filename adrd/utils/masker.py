from abc import ABC, abstractmethod
from functools import wraps
from typing import Any
from numpy.typing import NDArray
import numpy as np
from random import shuffle
from random import choice


class Masker(ABC):
    ''' ... '''
    def __init__(self,
        modalities: dict[str, dict[str, Any]],
    ) -> None:
        ''' ... '''
        self.modalities = modalities
    
    @abstractmethod
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
    ) -> dict[str, bool]:
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


class MissingMasker(Masker):
    ''' ... '''
    @Masker._keyerror_hint
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
    ) -> dict[str, bool]:
        ''' ... '''
        return {k: smp[k] is None for k in self.modalities}


class DropoutMasker(Masker):
    ''' ... '''
    @Masker._keyerror_hint
    def __init__(self, 
        modalities: dict[str, dict[str, Any]],
        dat: list[dict[str, int | NDArray[np.float32] | None]],
        dropout_rate: float = .5,
        dropout_strategy: str = 'permutation',
    ) -> None:
        ''' ... '''
        super().__init__(modalities)

        # allowed strategies for dropout
        assert dropout_strategy in ['simple', 'compensated', 'permutation']
        self.dropout_strategy = dropout_strategy

        # calculate missing rates
        missing_rates = {k: sum([dat[i][k] is None for i in range(len(dat))]) / len(dat) for k in modalities}

        # calculate dropout rates
        if dropout_strategy == 'simple':
            dropout_rates = {k: dropout_rate for k in modalities}

        elif dropout_strategy == 'compensated':
            dropout_rates = {k: (dropout_rate - missing_rates[k]) / (1 - missing_rates[k] + 1e-16) for k in modalities}
            dropout_rates = {k: 0 if dropout_rates[k] < 0 else dropout_rates[k] for k in modalities}

        # useful attributes
        if dropout_strategy != 'permutation':
            self.missing_rates = missing_rates
            self.dropout_rates = dropout_rates

    @Masker._keyerror_hint
    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
     ) -> dict[str, bool]:
        ''' ... '''
        if self.dropout_strategy == 'permutation':
            src_keys = [k for k in self.modalities if smp[k] is not None]
            shuffle(src_keys)
            src_keys = src_keys[:choice(range(1, len(src_keys) + 1))]
            mask = {k: True for k in self.modalities}
            for k in src_keys:
                mask[k] = False
            return mask

        else:
            # get missing mask first
            missing_mask = {k: smp[k] is None for k in self.modalities}

            # vectorize
            missing_mask_vec = np.array(list(missing_mask.values()))
            dropout_rate_vec = np.array(list(self.dropout_rates.values()))
            
            # generate dropout mask, at least 1 element shall be kept
            while True:
                dropout_mask_vec = np.random.rand(len(dropout_rate_vec)) < dropout_rate_vec
                dropout_mask_vec = dropout_mask_vec | missing_mask_vec
                if not np.all(dropout_mask_vec): break

            return {k: dropout_mask_vec[i] for i, k in enumerate(self.modalities.keys())}

class LabelMasker():
    ''' ... '''
    def __init__(self, 
        modalities: dict[str, dict[str, Any]],
    ) -> None:
        ''' ... '''

        # useful attributes
        self.modalities = modalities

    def __call__(self,
        smp: dict[str, int | NDArray[np.float32] | None],
     ) -> dict[str, int | NDArray[np.float32]]:
        ''' ... '''
        # get missing mask
        label_mask = {k: 1 if smp[k] is not None else 0 for k in self.modalities}
        # print(label_mask)

        return label_mask
