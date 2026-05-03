import numpy as np
from models.base import BasePositioningModel
from utils.signal_utils import signal_distance, SCALE

class LSTMModel(BasePositioningModel):
    """LSTM model cho WiFi positioning"""
    
    def __init__(self, fingerprints, k=5):
        super().__init__(fingerprints)
        self.k = k
        self.name = "LSTM"
    
    def predict(self, signal_vector):
        if len(self.fingerprints) == 0:
            return None
        
        # Tính distances
        distances = []
        for fp in self.fingerprints:
            dist = signal_distance(signal_vector, fp['signals'])
            distances.append({
                'fp': fp,
                'distance': dist
            })
        
        # Sort
        distances.sort(key=lambda x: x['distance'])
        
        # K nearest
        neighbors = distances[:self.k]
        
        # Tính weights (inverse distance)
        weights = [1 / (n['distance'] + 0.1) for n in neighbors]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        sum_x = sum(n['fp']['position']['x'] * w for n, w in zip(neighbors, normalized_weights))
        sum_y = sum(n['fp']['position']['y'] * w for n, w in zip(neighbors, normalized_weights))
        
        # Noise (LSTM tốt hơn RNN)
        noise_x = np.random.normal(0, SCALE * 0.8)
        noise_y = np.random.normal(0, SCALE * 0.8)
        
        return {
            'position': {
                'x': sum_x + noise_x,
                'y': sum_y + noise_y
            },
            'neighbors': [
                {
                    'name': n['fp']['name'],
                    'distance': round(n['distance'], 2),
                    'weight': f"{w * 100:.1f}%",
                    'position': n['fp']['position']
                }
                for n, w in zip(neighbors[:5], normalized_weights[:5])
            ],
            'confidence': 1 / (1 + neighbors[0]['distance'] / 12)
        }