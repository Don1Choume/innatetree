import sys
from pathlib import Path
import pickle

src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(src_dir))

from data import MatlabGen, TimingGen, HandWritingGen
from inlearn import InnateLearn

class BaseExperiment(object):
    def __init__(self, random_state=None):
        self.model = None
        self.inputs = None
        self.outputs = None
        self.random_state = random_state

    def train(self, n_loops_innate, n_loops_recurr, n_loops_readout):
        # timing: n_loops_innate=1, n_loops_recurr=20, n_loops_readout=10
        # handwriting: n_loops_innate=5, n_loops_recurr=30, n_loops_readout=10
        # create network and get innate trajectory for target
        for i_loop in range(n_loops_innate):
            self.model.fit(self.inputs, self.outputs, train_window=self.train_window,
                            keep_weight=False, keep_P=False,
                            train_recurr=False, train_readout=False, get_target_innate_X=True)
            print('innate learning {0} / {1} done.'.format(i_loop+1, n_loops_innate))

        # train recurrent
        for i_loop in range(n_loops_recurr):
            self.model.fit(self.inputs, self.outputs, train_window=self.train_window,
                            keep_weight=True, keep_P=True,
                            train_recurr=True, train_readout=False, get_target_innate_X=False)
            print('recurr learning {0} / {1} done.'.format(i_loop+1, n_loops_recurr))

        # train readout
        for i_loop in range(n_loops_readout):
            self.model.fit(self.inputs, self.outputs, train_window=self.train_window,
                            keep_weight=True, keep_P=True,
                            train_recurr=False, train_readout=True, get_target_innate_X=False)
            print('readout learning {0} / {1} done.'.format(i_loop+1, n_loops_readout))

    def predict(self, X=None):
        if X is None:
            X = self.inputs
        return self.model.predict(X)

    def load_data(self):
        raise NotImplementedError()

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)


class OriginalExperiment(BaseExperiment):
    def __init__(self, num_units=800, num_plastic_units=480, p_connect=0.1, g=1.5,
                 delta=1.0, noise_amplitude=0.001, dt=1, tau=10.0, learning_every=2,
                 save_history=False, copy_X=True, verbose=False, random_state=None):
        self.num_units = num_units
        self.num_plastic_units = num_plastic_units
        self.p_connect = p_connect
        self.g = g
        self.delta = delta
        self.noise_amplitude = noise_amplitude
        self.dt = dt
        self.tau = tau
        self.learning_every = learning_every
        self.save_history = save_history
        self.copy_X = copy_X
        self.verbose = verbose
        super().__init__(random_state)

        self.model = InnateLearn(
            num_units=self.num_units,
            num_plastic_units=self.num_plastic_units,
            p_connect=self.p_connect,
            g=self.g,
            delta=self.delta,
            noise_amplitude=self.noise_amplitude,
            dt=self.dt,
            tau=self.tau,
            learning_every=self.learning_every,
            save_history=self.save_history,
            copy_X=self.copy_X,
            verbose=self.verbose,
            random_state=self.random_state)

        self.data_gen = None

    def load_data(self, target='timing', **kwargs):
        dt = self.model.dt
        if target=='timing':
            self.data_gen = TimingGen(dt=dt, **kwargs)
        else:
            self.data_gen = HandWritingGen(dt=dt, **kwargs)
        self.inputs = self.data_gen.gen_input()
        self.outputs = self.data_gen.gen_output()
        self.train_window = self.data_gen.gen_train_window()


class AttractorTree(BaseExperiment):
    pass

if __name__ == '__main__':
    import numpy as np
    rand_state = np.random.RandomState(seed=42)
    OE = OriginalExperiment(verbose=True, random_state=rand_state)
    OE.load_data(target='timing')
    OE.train(n_loops_innate=1, n_loops_recurr=20, n_loops_readout=10)
    # OE.load_data(target='handwriting')
    # OE.train(n_loops_innate=5, n_loops_recurr=30, n_loops_readout=10)
    y = OE.predict()
    print(y.shape, y)
    # print(y)
    # pass