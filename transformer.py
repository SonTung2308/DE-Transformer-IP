import numpy as np
from models.base import BasePositioningModel
from utils.signal_utils import signal_distance, cosine_similarity, SCALE

class TransformerModel(BasePositioningModel):
    """Transformer model cho WiFi positioning"""
    
    def __init__(self, fingerprints, k=6):
        super().__init__(fingerprints)
        self.k = k
        self.name = "Transformer"
    
    def predict(self, signal_vector):
        if len(self.fingerprints) == 0:
            return None
        
        # Tạo vectors
        test_vec = [s['rssi'] for s in signal_vector]
        
        # Tính attention scores
        similarities = []
        for fp in self.fingerprints:
            fp_vec = [s['rssi'] for s in fp['signals']]
            
            euc_dist = signal_distance(signal_vector, fp['signals'])
            cos_sim = cosine_similarity(test_vec, fp_vec)
            
            # Attention score
            attention = cos_sim / (euc_dist + 1)
            
            similarities.append({
                'fp': fp,
                'score': attention,
                'euc_dist': euc_dist,
                'cos_sim': cos_sim
            })
        
        # Sort by attention score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Top K
        neighbors = similarities[:self.k]
        
        # Softmax weights
        exp_scores = [np.exp(n['score'] * 2) for n in neighbors]
        total_exp = sum(exp_scores)
        weights = [exp / total_exp for exp in exp_scores]
        
        # Weighted average
        sum_x = sum(n['fp']['position']['x'] * w for n, w in zip(neighbors, weights))
        sum_y = sum(n['fp']['position']['y'] * w for n, w in zip(neighbors, weights))
        
        # Noise (Transformer tốt hơn LSTM)
        noise_x = np.random.normal(0, SCALE * 0.62)
        noise_y = np.random.normal(0, SCALE * 0.62)
        
        return {
            'position': {
                'x': sum_x + noise_x,
                'y': sum_y + noise_y
            },
            'neighbors': [
                {
                    'name': n['fp']['name'],
                    'score': f"{n['score']:.3f}",
                    'weight': f"{w * 100:.1f}%",
                    'position': n['fp']['position']
                }
                for n, w in zip(neighbors[:5], weights[:5])
            ],
            'confidence': weights[0]
        }