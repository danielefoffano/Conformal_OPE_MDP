import numpy as np
from scipy.stats import bootstrap
from typing import Sequence
from types_cp import Trajectory, Point, Interval, ScoresWeightsData
from weights import WeightsEstimator
from numpy.typing import NDArray
from utils import filter_scores_weights
from policy import Policy

def evaluate_trajectory(test_point: Trajectory, weights_estimator: WeightsEstimator, data: ScoresWeightsData) -> Interval:
    pass
    
class ISMethod(object):
    def __init__(self, pi_target: Policy, horizon: int, discount: float, data_cal: Sequence[Trajectory], weights_estimator: WeightsEstimator):
        self._pi_target = pi_target
        self._horizon = horizon
        self._discount = discount
        self._data_cal = data_cal
        self._weights_estimator = weights_estimator
        self._results = {}

    def _estimate_confidence_set(self, values: Sequence[float], weights: Sequence[float]):
        ordered_values, ordered_weights = filter_scores_weights(values, weights, enable_fft=False)

        cdf = np.cumsum(ordered_weights) / np.sum(ordered_weights)

        res = ordered_values[np.argwhere(cdf >= 0.05)]
        q_low = res[0].item() if len(res) > 0 else ordered_values.min()
        
        res = ordered_values[np.argwhere(cdf >= 0.95)]
        q_high = res[0].item() if len(res) > 0 else ordered_values.max()
        return q_low, q_high
    
    def _estimate_bootstrap(self, values: NDArray[np.int64], initial_state: int, alpha: float):

        weights = np.array(list(map(lambda x: self._weights_estimator.evaluate_point(Point(initial_state, x)), values)))
        ordered_values, ordered_weights = filter_scores_weights(values, weights, enable_fft=False)

        cdf = np.cumsum(ordered_weights) / np.sum(ordered_weights)

        
        res = ordered_values[np.argwhere(cdf >= alpha)]
        ret = res[0].item() if len(res) > 0 else (ordered_values.min() if alpha < 0.5 else ordered_values.max())
        return ret
        
    def build_set(self, test_points: Sequence[Trajectory],n_cpu: int = 2) -> Sequence[Interval]:
        intervals: Sequence[Interval] = []
        
        for point in test_points:
            if point.initial_state not in self._results:
                trajectories = list(filter(lambda x: x.initial_state == point.initial_state, self._data_cal))
                values, weights = zip(*list(
                    map(lambda x: (x.cumulative_reward, self._weights_estimator.evaluate_point(
                                                                Point(x.initial_state, x.cumulative_reward))),
                        trajectories)))
    
                values = np.array(values)
                weights = np.array(weights)
    
                b_qlow = bootstrap(values[np.newaxis,:], lambda x: self._estimate_bootstrap(x, point.initial_state, 0.05), method='percentile',  vectorized=False)
                b_qhigh = bootstrap(values[np.newaxis,:], lambda x: self._estimate_bootstrap(x, point.initial_state, 0.95),  method='percentile',vectorized=False)
         
                qlow, qhigh = self._estimate_confidence_set(values, weights)
                
                b_qlow = qlow if np.isnan(b_qlow.confidence_interval.low) else (b_qlow.confidence_interval.low + b_qlow.confidence_interval.high)/2
                b_qhigh = qhigh if np.isnan(b_qhigh.confidence_interval.high) else (b_qhigh.confidence_interval.high + b_qhigh.confidence_interval.low)/2
                self._results[point.initial_state] = {
                    'values': values,
                    'weights': weights,
                    'qlow': b_qlow,
                    'qhigh': b_qhigh
                }
            
            intervals.append(Interval(Point(point.initial_state, point.cumulative_reward), qlow, qhigh,
                                      qlow, qhigh, qlow, qhigh, qlow, qhigh, [0], [0], [0], [0], [0]))
        
        
        return intervals
        
