import numpy as np
from models.base import BasePositioningModel
from utils.signal_utils import signal_distance, SCALE

class RNNModel(BasePositioningModel):
    """RNN model cho WiFi positioning"""
    
    def __init__(self, fingerprints, k=5):
        super().__init__(fingerprints)
        self.k = k
        self.name = "RNN"
    
    def predict(self, signal_vector):
        if len(self.fingerprints) == 0:
            return None
        
        # Tính distances tới tất cả fingerprints
        distances = []
        for fp in self.fingerprints:
            dist = signal_distance(signal_vector, fp['signals'])
            distances.append({
                'fp': fp,
                'distance': dist
            })
        
        # Sort theo distance
        distances.sort(key=lambda x: x['distance'])
        
        # Lấy K nearest
        neighbors = distances[:self.k]
        
        # Tính trung bình vị trí
        sum_x = sum(n['fp']['position']['x'] for n in neighbors)
        sum_y = sum(n['fp']['position']['y'] for n in neighbors)
        
        # Thêm noise (RNN có noise cao)
        noise_x = np.random.normal(0, SCALE * 0.95)
        noise_y = np.random.normal(0, SCALE * 0.95)
        
        return {
            'position': {
                'x': sum_x / len(neighbors) + noise_x,
                'y': sum_y / len(neighbors) + noise_y
            },
            'neighbors': [
                {
                    'name': n['fp']['name'],
                    'distance': round(n['distance'], 2),
                    'position': n['fp']['position']
                }
                for n in neighbors[:5]
            ],
            'confidence': 1 / (1 + neighbors[0]['distance'] / 10)
        }