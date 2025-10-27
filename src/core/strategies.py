import pickle
from typing import Union, List
from abc import ABC, abstractmethod

from collections import namedtuple
import mlflow 

class LearningModel(ABC):
    def __init__(self,model):
        self.model = model

    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def predict(data):
        pass
    @abstractmethod
    def hp_tunning(self,**kwargs):
        pass

    @abstractmethod
    def save_model(self,path:str):
        pass
