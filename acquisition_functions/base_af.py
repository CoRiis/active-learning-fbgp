# Base for Acquisition Function (af)
import abc


class BaseAcquisitionFunction(abc.ABC):
    def __init__(self, model, candidate_points):
        self.model = model
        self.candidate_points = candidate_points

    @abc.abstractmethod
    def evaluate(self, x):
        pass
