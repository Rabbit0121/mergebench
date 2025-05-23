# Model Soups
import copy
from typing import Dict, List

class ModelSoups():
    def __init__(self, checkpoint_list:List[Dict]):
        super().__init__()
        self.checkpoint_list = checkpoint_list
        self.alpha = 1 / (len(self.checkpoint_list))
    
    def run(self):
        uniform_soup = {k : v * self.alpha for k, v in self.checkpoint_list[0].items()}
        # uniform_soup = {k : v  for k, v in self.checkpoint_list[0].items()}
        for checkpoint in self.checkpoint_list[1:]:
            uniform_soup = {k : v * self.alpha + uniform_soup[k] for k, v in checkpoint.items()}
            
        return uniform_soup
