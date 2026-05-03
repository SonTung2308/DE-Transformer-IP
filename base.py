from abc import ABC, abstractmethod
import numpy as np
from utils.signal_utils import signal_distance, SCALE

class BasePositioningModel(ABC):
    """Base class cho tất cả positioning models"""
    
    def __init__(self, fingerprints):
        self.fingerprints = fingerprints
        self.name = "Base"
    
    @abstractmethod
    def predict(self, signal_vector):
        """Dự đoán vị trí dựa trên signal vector"""
        pass
    
    def predict_batch(self, signal_vectors):
        """Dự đoán nhiều signals"""
        return [self.predict(sv) for sv in signal_vectors]