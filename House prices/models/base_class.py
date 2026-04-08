from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def train(self, X, y, folds = None):
        pass

    @abstractmethod
    def predict(self, X):
        pass