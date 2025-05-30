
from verl.utils.reward_score import math

def compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    return math.compute_score(solution_str, ground_truth)
