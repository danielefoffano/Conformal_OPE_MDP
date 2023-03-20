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

class ScoresWeightsData(NamedTuple):
    scores: NDArray[np.float64]
    weights: NDArray[np.float64]
    
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
    coverage_double: float
    coverage_cumul: float
    avg_length: float
    std_length: float
    
    avg_length_double: float
    std_length_double: float
    
    avg_length_cumul: float
    std_length_cumul: float
    lower_vals_behavior: NDArray[np.float64]
    upper_val_behavior: NDArray[np.float64]
    avg_quantiles: NDArray[np.float64]
    std_quantiles: NDArray[np.float64]
    
    avg_quantiles_double_low: NDArray[np.float64]
    std_quantiles_double_low: NDArray[np.float64]
    avg_quantiles_double_high: NDArray[np.float64]
    std_quantiles_double_high: NDArray[np.float64]
    
    avg_quantiles_cumul_low: NDArray[np.float64]
    std_quantiles_cumul_low: NDArray[np.float64]
    avg_quantiles_cumul_high: NDArray[np.float64]
    std_quantiles_cumul_high: NDArray[np.float64]
    
    lower_vals_target: NDArray[np.float64]
    upper_vals_target: NDArray[np.float64]
    lower_vals_target_double: NDArray[np.float64]
    upper_vals_target_double: NDArray[np.float64]
    lower_vals_target_cumul: NDArray[np.float64]
    upper_vals_target_cumul: NDArray[np.float64]

class Interval(NamedTuple):
    point: Point
    min_y: float
    max_y: float
    min_y_double: float
    max_y_double: float
    min_y_cumul: float
    max_y_cumul: float
    lower_val: float
    upper_val: float
    quantiles: Sequence[float]
    quantiles_double_low: Sequence[float]
    quantiles_double_high: Sequence[float]
    quantiles_double_cumul_low: Sequence[float]
    quantiles_double_cumul_high: Sequence[float]
    
    @staticmethod
    def analyse_intervals(intervals: Sequence[Interval]) -> TestResults:
        included = 0
        included_double = 0
        included_cumul = 0
        
        
        lengths = np.zeros(len(intervals))
        lengths_double = np.zeros(len(intervals))
        lengths_cumul = np.zeros(len(intervals))
        lower_vals = np.zeros(len(intervals))
        upper_vals = np.zeros(len(intervals))
        std_quantile_vals = np.zeros(len(intervals))
        avg_quantile_vals = np.zeros(len(intervals))
        
        std_quantile_double_low_vals = np.zeros(len(intervals))
        avg_quantile_double_low_vals = np.zeros(len(intervals))
        
        std_quantile_double_high_vals = np.zeros(len(intervals))
        avg_quantile_double_high_vals = np.zeros(len(intervals))
        
        std_quantile_cumul_low_vals = np.zeros(len(intervals))
        avg_quantile_cumul_low_vals = np.zeros(len(intervals))
        
        std_quantile_cumul_high_vals = np.zeros(len(intervals))
        avg_quantile_cumul_high_vals = np.zeros(len(intervals))
        
        
        min_y = np.zeros(len(intervals))
        max_y = np.zeros(len(intervals))
        min_y_double = np.zeros(len(intervals))
        max_y_double = np.zeros(len(intervals))
        min_y_cumul = np.zeros(len(intervals))
        max_y_cumul = np.zeros(len(intervals))
        
        for idx, interval in enumerate(intervals):
            point = interval.point
            if point.cumulative_reward >= interval.min_y and point.cumulative_reward <= interval.max_y:
                included += 1
                
            if point.cumulative_reward >= interval.min_y_double and point.cumulative_reward <= interval.max_y_double:
                included_double += 1
                
            if point.cumulative_reward >= interval.min_y_cumul and point.cumulative_reward <= interval.max_y_cumul:
                included_cumul += 1
                
            lengths[idx] = interval.max_y - interval.min_y
            lengths_double[idx] = interval.max_y_double - interval.min_y_double
            lengths_cumul[idx] = interval.max_y_cumul - interval.min_y_cumul
            lower_vals[idx] = interval.lower_val
            upper_vals[idx] = interval.upper_val
            avg_quantile_vals[idx] = np.mean(interval.quantiles)
            std_quantile_vals[idx] = np.std(interval.quantiles, ddof=1)
            
            avg_quantile_double_low_vals[idx] = np.mean(interval.quantiles_double_low)
            std_quantile_double_low_vals[idx] = np.std(interval.quantiles_double_low, ddof=1)
            
            avg_quantile_double_high_vals[idx] = np.mean(interval.quantiles_double_high)
            std_quantile_double_high_vals[idx] = np.std(interval.quantiles_double_high, ddof=1)
            
            avg_quantile_cumul_low_vals[idx] = np.mean(interval.quantiles_double_cumul_low)
            std_quantile_cumul_low_vals[idx] = np.std(interval.quantiles_double_cumul_low, ddof=1)
            
            avg_quantile_cumul_high_vals[idx] = np.mean(interval.quantiles_double_cumul_high)
            std_quantile_cumul_high_vals[idx] = np.std(interval.quantiles_double_cumul_high, ddof=1)
            
            min_y[idx] = interval.min_y
            max_y[idx] = interval.max_y
            min_y_double[idx] = interval.min_y_double
            max_y_double[idx] = interval.max_y_double
            min_y_cumul[idx] = interval.min_y_cumul
            max_y_cumul[idx] = interval.max_y_cumul
        
        coverage = included / len(intervals)
        coverage_double = included_double / len(intervals)
        coverage_cumul = included_cumul / len(intervals)
        return TestResults(coverage * 100, coverage_double * 100, coverage_cumul*100,
                           lengths.mean(), lengths.std(ddof=1), 
                           lengths_double.mean(), lengths_double.std(ddof=1),
                           lengths_cumul.mean(), lengths_cumul.std(ddof=1),
                           lower_vals,
                           upper_vals, avg_quantile_vals, std_quantile_vals,
                           avg_quantile_double_low_vals, std_quantile_double_low_vals,
                           avg_quantile_double_high_vals, std_quantile_double_high_vals,
                           
                           avg_quantile_cumul_low_vals, std_quantile_cumul_low_vals,
                           avg_quantile_cumul_high_vals, std_quantile_cumul_high_vals,
                           min_y, max_y, min_y_double, max_y_double, min_y_cumul, max_y_cumul )


class LoggerResults(NamedTuple):
    epsilon: float
    coverage: float
    coverage_double: float
    coverage_cumul: float
    avg_length: float
    std_length: float
    
    
    avg_length_double: float
    std_length_double: float
    
    avg_length_cumul: float
    std_length_cumul: float
    avg_interval_target_lower: float
    avg_interval_target_upper: float
    std_interval_target_lower: float
    std_interval_target_upper: float
    
    avg_interval_target_double_lower: float
    avg_interval_target_double_upper: float
    std_interval_target_double_lower: float
    std_interval_target_double_upper: float
    
    avg_interval_target_cumul_lower: float
    avg_interval_target_cumul_upper: float
    std_interval_target_cumul_lower: float
    std_interval_target_cumul_upper: float
    
    avg_interval_behavior_lower: float
    avg_interval_behavior_upper: float
    std_interval_behavior_lower: float
    std_interval_behavior_upper: float
    avg_quantile: float
    avg_std_quantile: float
    std_avg_quantile: float
    
    avg_double_quantile_low: float
    avg_std_double_quantile_low: float
    std_avg_double_quantile_low: float
    
    avg_double_quantile_high: float
    avg_std_double_quantile_high: float
    std_avg_double_quantile_high: float
    
    
    avg_cumul_quantile_low: float
    avg_std_cumul_quantile_low: float
    std_avg_cumul_quantile_low: float
    
    avg_cumul_quantile_high: float
    avg_std_cumul_quantile_high: float
    std_avg_cumul_quantile_high: float
    
    
    
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
