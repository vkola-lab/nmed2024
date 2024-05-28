import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from functools import lru_cache
from functools import cached_property
from typing import Self, Any
from pickle import dump
from pickle import load
from abc import ABC, abstractmethod

from . import ADRDModel
from ..utils import Formatter
from ..utils import MissingMasker


def calibration_curve(
    y_true: list[int],
    y_pred: list[float],
    n_bins: int = 10,
    ratio: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Compute true and predicted probabilities for a calibration curve. The method
    assumes the inputs come from a binary classifier, and discretize the [0, 1] 
    interval into bins.

    Note that this function is an alternative to
    sklearn.calibration.calibration_curve() which can only estimate the absolute
    proportion of positive cases in each bin.

    Parameters
    ----------
    y_true : list[int]
        True targets.
    y_pred : list[float]
        Probabilities of the positive class.
    n_bins : int, default=10
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without corresponding
        values in y_prob) will not be returned, thus the returned arrays may
        have less than n_bins values.
    ratio : float, default=1.0
        Used to adjust the class balance.

    Returns
    -------
    prob_true : list[float]
        The proportion of positive samples in each bin.
    prob_pred : list[float]
        The mean predicted probability in each bin.
    """
    # generate "n_bin" intervals
    tmp = np.around(np.linspace(0, 1, n_bins + 1), decimals=6)
    intvs = [(tmp[i - 1], tmp[i]) for i in range(1, len(tmp))]
    
    # pair up (pred, true) and group them by intervals
    tmp = list(zip(y_pred, y_true))
    intv_pairs = {(l, r): [p for p in tmp if l <= p[0] < r] for l, r in intvs}

    # calculate balanced proportion of POSITIVE cases for each intervel
    # along with the balanced averaged predictions
    intv_prob_true: dict[tuple, float] = dict()
    intv_prob_pred: dict[tuple, float] = dict()
    for intv, pairs in intv_pairs.items():
        # number of cases that fall into the interval
        n_pairs = len(pairs)

        # it's likely that no predictions fall into the interval
        if n_pairs == 0: continue

        # count number of positives and negatives in the interval
        n_pos = sum([p[1] for p in pairs])
        n_neg = n_pairs - n_pos

        # calculate adjusted proportion of positives
        intv_prob_true[intv] = n_pos / (n_pos + n_neg * ratio)

        # calculate adjusted avg. predictions
        sum_pred_pos = sum([p[0] for p in pairs if p[1] == 1])
        sum_pred_neg = sum([p[0] for p in pairs if p[1] == 0])
        intv_prob_pred[intv] = (sum_pred_pos + sum_pred_neg * ratio)
        intv_prob_pred[intv] /= (n_pos + n_neg * ratio)

    prob_true = list(intv_prob_true.values())
    prob_pred = list(intv_prob_pred.values())
    return prob_true, prob_pred


class CalibrationCore(BaseEstimator):
    """
    A wrapper class of multiple regressors to predict the proportions of
    positive samples from the predicted probabilities. The method for
    calibration can be 'sigmoid' which corresponds to Platt's method (i.e. a 
    logistic regression model) or 'isotonic' which is a non-parametric approach.
    It is not advised to use isotonic calibration with too few calibration
    samples (<<1000) since it tends to overfit.

    TODO
    ----
    - 'sigmoid' method is not trivial to implement.
    """
    def __init__(self, 
        method: str = 'isotonic',
    ) -> None:
        """
        Initialization function of CalibrationCore class.

        Parameters
        ----------
        method : {'sigmoid', 'isotonic'}, default='isotonic'
            The method to use for calibration. can be 'sigmoid' which
            corresponds to Platt's method (i.e. a logistic regression model) or
            'isotonic' which is a non-parametric approach. It is not advised to
            use isotonic calibration with too few calibration samples (<<1000)
            since it tends to overfit.

        Raises
        ------
        ValueError
            Sigmoid approach has not been implemented.
        """        
        assert method in ('sigmoid', 'isotonic')
        if method == 'sigmoid':
            raise ValueError('Sigmoid approach has not been implemented.')
        self.method = method

    def fit(self, 
        prob_pred: list[float], 
        prob_true: list[float],
    ) -> Self:
        """
        Fit the underlying regressor using prob_pred, prob_true as training
        data.

        Parameters
        ----------
        prob_pred : list[float]
            Probabilities predicted directly by a model.
        prob_true : list[float]
            Target probabilities to calibrate to.

        Returns
        -------
        Self
            CalibrationCore object.
        """              
        # using Platt's method for calibration
        if self.method == 'sigmoid':
            self.model_ = LogisticRegression()
            self.model_.fit(prob_pred, prob_true)

        # using isotonic calibration
        elif self.method == 'isotonic':
            self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.model_.fit(prob_pred, prob_true)

        return self

    def predict(self,
        prob_pred: list[float],
    ) -> list[float]:
        """
        Calibrate the input probabilities using the fitted regressor.

        Parameters
        ----------
        prob_pred : list[float]
            Probabilities predicted directly by a model.

        Returns
        -------
        prob_cali : list[float]
            Calibrated probabilities.
        """        
        # as usual, the core needs to be fitted
        check_is_fitted(self)

        # note that logistic regression is classification model, we need to call
        # 'predict_proba' instead of 'predict' to get the calibrated results
        if self.method == 'sigmoid':
            prob_cali = self.model_.predict_proba(prob_pred)
        elif self.method == 'isotonic':
            prob_cali = self.model_.predict(prob_pred)

        return prob_cali
    

class CalibratedClassifier(ABC):
    """
    Abstract class of calibrated classifier.
    """
    def __init__(self, 
        model: ADRDModel,
        background_src: list[dict[str, Any]],
        background_tgt: list[dict[str, Any]],
        background_is_embedding: dict[str, bool] | None = None,
        method: str = 'isotonic',
    ) -> None:
        """
        Constructor of Calibrator class.

        Parameters
        ----------
        model : ADRDModel
            Fitted model to calibrate.
        background_src : list[dict[str, Any]]
            Features of the background dataset.
        background_tgt : list[dict[str, Any]]
            Labels of the background dataset.
        method : {'sigmoid', 'isotonic'}, default='isotonic'
            Method used by the underlying regressor. 
        """
        self.method = method
        self.model = model
        self.src_modalities = model.src_modalities
        self.tgt_modalities = model.tgt_modalities
        self.background_is_embedding = background_is_embedding

        # format background data
        fmt_src = Formatter(self.src_modalities)
        fmt_tgt = Formatter(self.tgt_modalities)
        self.background_src = [fmt_src(smp) for smp in background_src]
        self.background_tgt = [fmt_tgt(smp) for smp in background_tgt]
    
    @abstractmethod
    def predict_proba(self, 
        src: list[dict[str, Any]],
        is_embedding: dict[str, bool] | None = None,
    ) -> list[dict[str, float]]:
        """
        This method returns calibrated probabilities of classification.

        Parameters
        ----------
        src : list[dict[str, Any]]
            Features of the input samples.

        Returns
        -------
        list[dict[str, float]]
            Calibrated probabilities.
        """ 
        pass

    def predict(self,
        src: list[dict[str, Any]],
        is_embedding: dict[str, bool] | None = None,
    ) -> list[dict[str, int]]:
        """
        Make predictions based on the results of predict_proba().

        Parameters
        ----------
        x : list[dict[str, Any]]
            Input features.

        Returns
        -------
        list[dict[str, int]]
            Calibrated predictions.
        """
        proba = self.predict_proba(src, is_embedding)
        return [{k: int(smp[k] > 0.5) for k in self.tgt_modalities} for smp in proba]

    def save(self,
        filepath_state_dict: str,
    ) -> None:
        """
        Save the state dict and the underlying model to the given paths.

        Parameters
        ----------
        filepath_state_dict : str
            File path to save the state_dict which includes the background
            dataset and the regressor information.
        filepath_wrapped_model : str | None, default=None
            File path to save the wrapped model. If None, the model won't be
            saved. 
        """
        # save state dict
        state_dict = {
            'background_src': self.background_src,
            'background_tgt': self.background_tgt,
            'background_is_embedding': self.background_is_embedding,
            'method': self.method,
        }
        with open(filepath_state_dict, 'wb') as f:
            dump(state_dict, f)

    @classmethod
    def from_ckpt(cls,
        filepath_state_dict: str,
        filepath_wrapped_model: str,
    ) -> Self:
        """
        Alternative constructor which loads from checkpoint.

        Parameters
        ----------
        filepath_state_dict : str
            File path to load the state_dict which includes the background
            dataset and the regressor information.
        filepath_wrapped_model : str
            File path of the wrapped model.

        Returns
        -------
        Self
            CalibratedClassifier class object.
        """
        with open(filepath_state_dict, 'rb') as f:
            kwargs = load(f)
        kwargs['model'] = ADRDModel.from_ckpt(filepath_wrapped_model)
        return cls(**kwargs)


class DynamicCalibratedClassifier(CalibratedClassifier):
    """
    The dynamic approach generates background predictions based on the
    missingness pattern of each input. With an astronomical number of
    missingness patterns, calibrating each sample requires a comprehensive
    process that involves running the ADRDModel on the majority of the
    background data and training a corresponding regressor. This results in a
    computationally intensive calculation.
    """
    def predict_proba(self,
        src: list[dict[str, Any]],
        is_embedding: dict[str, bool] | None = None,
    ) -> list[dict[str, float]]:
        
        # initialize mask generator and format inputs
        msk_gen = MissingMasker(self.src_modalities)
        fmt_src = Formatter(self.src_modalities)
        src = [fmt_src(smp) for smp in src]

        # calculate calibrated probabilities
        calibrated_prob: list[dict[str, float]] = []
        for smp in src:
            # model output and missingness pattern
            prob = self.model.predict_proba([smp], is_embedding)[0]
            mask = tuple(msk_gen(smp).values())

            # get/fit core and calculate calibrated probabilities
            core = self._fit_core(mask)
            calibrated_prob.append({k: core[k].predict([prob[k]])[0] for k in self.tgt_modalities})

        return calibrated_prob
    
    # @lru_cache(maxsize = None)
    def _fit_core(self,
        missingness_pattern: tuple[bool],
    ) -> dict[str, CalibrationCore]:
        ''' ... ''' 
        # remove features from all background samples accordingly
        background_src, background_tgt = [], []
        for src, tgt in zip(self.background_src, self.background_tgt):
            src = {k: v for j, (k, v) in enumerate(src.items()) if missingness_pattern[j] == False}

            # make sure there is at least one feature available
            if len([v is not None for v in src.values()]) == 0: continue
            background_src.append(src)
            background_tgt.append(tgt)

        # run model on background samples and collection predictions
        background_prob = self.model.predict_proba(background_src, self.background_is_embedding, _batch_size=1024)

        # list[dict] -> dict[list]
        N = len(background_src)
        background_prob = {k: [background_prob[i][k] for i in range(N)] for k in self.tgt_modalities}
        background_true = {k: [background_tgt[i][k] for i in range(N)] for k in self.tgt_modalities}

        # now, fit cores
        core: dict[str, CalibrationCore] = dict()
        for k in self.tgt_modalities:
            prob_true, prob_pred = calibration_curve(
                background_true[k], background_prob[k],
                ratio = self.background_ratio[k],
            )
            core[k] = CalibrationCore(self.method).fit(prob_pred, prob_true)
        
        return core
    
    @cached_property
    def background_ratio(self) -> dict[str, float]:
        ''' The ratio of positives over negatives in the background dataset. '''
        return {k: self.background_n_pos[k] / self.background_n_neg[k] for k in self.tgt_modalities}

    @cached_property
    def background_n_pos(self) -> dict[str, int]:
        ''' Number of positives w.r.t each target in the background dataset. '''
        return {k: sum([d[k] for d in self.background_tgt]) for k in self.tgt_modalities}

    @cached_property
    def background_n_neg(self) -> dict[str, int]:
        ''' Number of negatives w.r.t each target in the background dataset. '''
        return {k: len(self.background_tgt) - self.background_n_pos[k] for k in self.tgt_modalities}


class StaticCalibratedClassifier(CalibratedClassifier):
    """
    The static approach generates background predictions without considering the
    missingness patterns.
    """
    def predict_proba(self,
        src: list[dict[str, Any]],
        is_embedding: dict[str, bool] | None = None,
    ) -> list[dict[str, float]]:

        # number of input samples
        N = len(src)

        # format inputs, and run ADRDModel, and convert to dict[list]
        fmt_src = Formatter(self.src_modalities)
        src = [fmt_src(smp) for smp in src]
        prob = self.model.predict_proba(src, is_embedding)
        prob = {k: [prob[i][k] for i in range(N)] for k in self.tgt_modalities}

        # calibrate probabilities
        core = self._fit_core()
        calibrated_prob = {k: core[k].predict(prob[k]) for k in self.tgt_modalities}

        # convert back to list[dict]
        calibrated_prob: list[dict[str, float]] = [
            {k: calibrated_prob[k][i] for k in self.tgt_modalities} for i in range(N)
        ]
        return calibrated_prob
    
    @lru_cache(maxsize = None)
    def _fit_core(self) -> dict[str, CalibrationCore]:
        ''' ... '''
        # run model on background samples and collection predictions
        background_prob = self.model.predict_proba(self.background_src, self.background_is_embedding, _batch_size=1024)

        # list[dict] -> dict[list]
        N = len(self.background_src)
        background_prob = {k: [background_prob[i][k] for i in range(N)] for k in self.tgt_modalities}
        background_true = {k: [self.background_tgt[i][k] for i in range(N)] for k in self.tgt_modalities}

        # now, fit cores
        core: dict[str, CalibrationCore] = dict()
        for k in self.tgt_modalities:
            prob_true, prob_pred = calibration_curve(
                background_true[k], background_prob[k],
                ratio = 1.0,
            )
            core[k] = CalibrationCore(self.method).fit(prob_pred, prob_true)
        
        return core