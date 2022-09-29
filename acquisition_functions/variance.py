from acquisition_functions.base_af import BaseAcquisitionFunction


class Variance(BaseAcquisitionFunction):
    def __init__(self, model, candidate_points, variance):
        super().__init__(model, candidate_points)

        self.variance = variance

    def evaluate(self, x):
        return self.variance
