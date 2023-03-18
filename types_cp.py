from __future__ import annotations
from typing import NamedTuple, Sequence
import numpy as np
from numpy.typing import NDArray

class Experience(NamedTuple):
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    
    
class Trajectory(NamedTuple):
    initial_state: int
    cumulative_reward: float
    trajectory: Sequence[Experience]

class Point(NamedTuple):
    """Object that contains an initial state and the cumulative reward
    """    
    initial_state: int
    cumulative_reward: float

class TestResults(NamedTuple):
    coverage: float
    avg_length: float
    std_length: float
    lower_vals_behavior: NDArray[np.float64]
    upper_val_behavior: NDArray[np.float64]
    avg_quantiles: NDArray[np.float64]
    std_quantiles: NDArray[np.float64]
    lower_vals_target: NDArray[np.float64]
    upper_vals_target: NDArray[np.float64]

class Interval(NamedTuple):
    point: Point
    min_y: float
    max_y: float
    lower_val: float
    upper_val: float
    quantiles: Sequence[float]
    
    @staticmethod
    def analyse_intervals(intervals: Sequence[Interval]) -> TestResults:
        included = 0
        lengths = np.zeros(len(intervals))
        lower_vals = np.zeros(len(intervals))
        upper_vals = np.zeros(len(intervals))
        std_quantile_vals = np.zeros(len(intervals))
        avg_quantile_vals = np.zeros(len(intervals))
        min_y = np.zeros(len(intervals))
        max_y = np.zeros(len(intervals))
        
        for idx, interval in enumerate(intervals):
            point = interval.point
            if point.cumulative_reward >= interval.min_y and point.cumulative_reward <= interval.max_y:
                included += 1
            lengths[idx] = interval.max_y - interval.min_y
            lower_vals[idx] = interval.lower_val
            upper_vals[idx] = interval.upper_val
            avg_quantile_vals[idx] = np.mean(interval.quantiles)
            std_quantile_vals[idx] = np.std(interval.quantiles, ddof=1)
            min_y[idx] = interval.min_y
            max_y[idx] = interval.max_y
        
        coverage = included / len(intervals)
        return TestResults(coverage * 100, lengths.mean(), lengths.std(ddof=1), lower_vals,
                           upper_vals, avg_quantile_vals, std_quantile_vals, min_y, max_y)


class LoggerResults(NamedTuple):
    epsilon: float
    coverage: float
    avg_length: float
    std_length: float
    avg_interval_target_lower: float
    avg_interval_target_upper: float
    std_interval_target_lower: float
    std_interval_target_upper: float
    
    avg_interval_behavior_lower: float
    avg_interval_behavior_upper: float
    std_interval_behavior_lower: float
    std_interval_behavior_upper: float
    avg_quantile: float
    avg_std_quantile: float
    std_avg_quantile: float
    horizon: int
    epsilon_pi_behavior: float
    avg_weights: float
    std_weights: float
    avg_ratio_what_w: float
    std_ratio_what_w: float
    median_ratio_what_w: float
    avg_delta_w: float
    std_delta_w: float
    median_delta_w: float
