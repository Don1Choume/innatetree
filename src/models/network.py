import numpy as np

class Network(object):
    def __init__(self, num_units, num_inputs, num_outputs, p_connect, scale, seed=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.p_connect = p_connect
        self.scale = scale
        np.random.seed(seed=seed)

        self.WXX_mask = np.random.choice([0, 1],
                                        size=(self.num_units, self.num_units), 
                                        p=[1-self.p_connect, self.p_connect])
        self.WXX = np.random.randn(self.num_units, self.num_units)*self.scale

        