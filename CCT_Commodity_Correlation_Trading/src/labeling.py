"""
Triple Barrier Labeling
Based on Advances in Financial Machine Learning methodology
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    def __init__(self, profit_take=0.02, stop_loss=0.01, time_limit=10):
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        self.time_limit = time_limit
        
    def label_events(self, data, features):
        """Apply triple barrier method"""
        logger.info("Applying triple barrier labeling...")
        
        labels = {}
        
        for pair_name, feat_df in features.items():
            eq_ret = feat_df['eq_ret'].values
            
            lbls = []
            for i in range(len(eq_ret) - self.time_limit):
                future_rets = eq_ret[i+1:i+self.time_limit+1]
                cum_ret = np.cumsum(future_rets)
                
                # Check barriers
                profit_hit = np.where(cum_ret >= self.profit_take)[0]
                loss_hit = np.where(cum_ret <= -self.stop_loss)[0]
                
                if len(profit_hit) > 0 and (len(loss_hit) == 0 or profit_hit[0] < loss_hit[0]):
                    label = 1  # Profit
                    time_to_hit = profit_hit[0]
                elif len(loss_hit) > 0:
                    label = -1  # Loss
                    time_to_hit = loss_hit[0]
                else:
                    label = 0  # Timeout
                    time_to_hit = self.time_limit
                    
                lbls.append({"label": label, "horizon": time_to_hit})
            
            labels[pair_name] = pd.DataFrame(lbls, index=feat_df.index[:len(lbls)])
        
        logger.info(f"Labeled {sum(len(l) for l in labels.values())} events")
        return labels
