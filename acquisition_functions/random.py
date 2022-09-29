import torch
from acquisition_functions.base_af import BaseAcquisitionFunction


class Random(BaseAcquisitionFunction):
    def __init__(self, model, candidate_points, args):
        super().__init__(model, candidate_points)
        self.args = args

    def evaluate(self, x):
        # Generate a random number for each sample in the pool
        return torch.rand([self.candidate_points.shape[0]])
