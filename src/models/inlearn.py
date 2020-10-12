import numpy as np
# from scipy.sparse import isspmatrix, csc_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted
from tqdm import tqdm

# "Robust Timing and Motor Patterns by Taming Chaos in Recurrent Neural Networks"
# Rodrigo Laje & Dean V. Buonomano 2013
# inspired by suppremental MATLAB code

class InnateLearn(BaseEstimator):
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
        self.random_state = random_state

    # more simplification must be necessary(by dividing into functions)
    def fit(self, X, y, train_window=None, keep_weight=False, keep_P=False,
                train_recurr=True, train_readout=True, get_target_innate_X=False):
        random_state = check_random_state(self.random_state)
        self._scale = self.g/np.sqrt(self.p_connect*self.num_units)

        X, y = check_X_y(X, y, copy=self.copy_X, multi_output=True)

        if y.shape[0] == 0:
            raise ValueError("y has 0 samples: %r" % y)

        self._num_inputs = X.shape[1]
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
            self._num_outputs = 1
        else:
            self._num_outputs = y.shape[1]

        if train_window is None:
            train_window = [True]*y.shape[0]
        elif len(train_window) != y.shape[0]:
            raise ValueError("train_window has diffenrent length to y: %r" % train_window)

        if train_recurr and not hasattr(self, '_Target_innate_X'):
            self.fit(X, y, train_window=train_window, keep_weight=keep_weight, keep_P=keep_P,
                train_recurr=False, train_readout=False, get_target_innate_X=True)

        n_steps = X.shape[0]

        if get_target_innate_X:
            noise_amp = 0
        else:
            noise_amp = self.noise_amplitude

        if (not keep_weight) or (not hasattr(self, '_WXX')):
            ## initialize internal weight
            # random sparse recurrent matrix between units.
            # indices in WXX are defined as WXX(postsyn,presyn),
            # that is WXX[i,j] = connection from Xi[j] onto Xi[i]
            # then the current into the postsynaptic unit is simply
            # (post)X_current = np.dot(WXX, (pre)Xi)
            WXX_mask = random_state.choice([0, 1], size=(self.num_units, self.num_units), 
                                                p=[1-self.p_connect, self.p_connect])
            WXX_temp = random_state.randn(self.num_units, self.num_units)*self._scale*WXX_mask
            np.fill_diagonal(WXX_temp, 0)
            self._WXX = WXX_temp
            self._WXX_ini = self._WXX

            # input connections WInputX(postsyn,presyn)
            self._WInputX = random_state.randn(self.num_units, self._num_inputs)

            # output connections WXOut(postsyn,presyn)
            self._WXOut = random_state.randn(self._num_outputs, self.num_units)/\
                            np.sqrt(self.num_units)
            self._WXOut_ini = self._WXOut

        if train_recurr and not (keep_P and hasattr(self, '_P_recurr')):
            # list of all recurrent units subject to plasticity
            plastic_units = list(range(self.num_plastic_units))
            # one P matrix for each plastic unit in the network
            # RLS: P matrix initialization
            # list of all units presynaptic to plastic_units
            self._pre_plastic_units = [self._WXX[plastic_units[i], :].nonzero()[0]
                                        for i in range(self.num_plastic_units)]
            self._P_recurr = [(1.0/self.delta)*np.eye(len(pre_punit))
                                 for pre_punit in self._pre_plastic_units]
        if train_readout and not (keep_P and hasattr(self, '_P_readout')):
            # one P matrix for each readout unit
            # RLS: P matrix initialization
            self._P_readout = np.tile((1.0/self.delta)*np.eye(self.num_units), 
                                             (self._num_outputs, 1, 1))

        if self.save_history or get_target_innate_X:
            self._X_history = np.zeros((n_steps, self.num_units))

        if self.save_history:
            self._Out_history = np.zeros((n_steps, self._num_outputs))
            # auxiliary variables
            self._WXOut_len = np.zeros((n_steps, 1))
            self._WXX_len = np.zeros((n_steps, 1))
            self._dW_readout_len = np.zeros((n_steps, 1))
            self._dW_recurr_len = np.zeros((n_steps, 1))

        # initial conditions
        Xv = 2*random_state.rand(self.num_units, 1)-1
        Xi = np.tanh(Xv)
        Out = np.zeros((self._num_outputs, 1))

        if self.verbose:
            bar = tqdm(total = n_steps)

        # integration loop
        for i in range(n_steps):
            Input = X[i,:].T[:, np.newaxis]

            # update units
            noise = noise_amp*random_state.randn(self.num_units, 1)*np.sqrt(self.dt)
            Xv_current = np.dot(self._WXX, Xi) + np.dot(self._WInputX, Input) + noise
            Xv = Xv + ((-Xv + Xv_current)/self.tau)*self.dt
            Xi = np.tanh(Xv)
            Out = np.dot(self._WXOut, Xi)

            # training
            if train_window[i] & (i%self.learning_every==0):
                if train_recurr:
                    # train recurrent
                    error = Xi - self._Target_innate_X[i, :].T[:, np.newaxis]
                    for plas in range(self.num_plastic_units):
                        X_pre_plastic = Xi[self._pre_plastic_units[plas]]
                        P_recurr_old = self._P_recurr[plas]
                        P_recurr_old_X = np.dot(P_recurr_old, X_pre_plastic)
                        den_recurr = 1 + np.dot(X_pre_plastic.T, P_recurr_old_X)
                        self._P_recurr[plas] = P_recurr_old - np.dot(P_recurr_old_X, P_recurr_old_X.T)/den_recurr
                        # update network matrix
                        dW_recurr = -np.dot(error[plas], (P_recurr_old_X/den_recurr).T)
                        self._WXX[plas, self._pre_plastic_units[plas]] = self._WXX[plas, self._pre_plastic_units[plas]] + dW_recurr
                        if self.save_history:
                            # store change in weights
                            self._dW_recurr_len[i] += np.sqrt(np.dot(dW_recurr, dW_recurr.T))

                if train_readout:
                    for out in range(self._num_outputs):
                        P_readout_old = np.squeeze(self._P_readout[out, :, :])
                        P_readout_old_X = np.dot(P_readout_old, Xi)
                        den_readout = 1 + np.dot(Xi.T, P_readout_old_X)
                        self._P_readout[out,:,:] = P_readout_old - np.dot(P_readout_old_X, P_readout_old_X.T)/den_readout
                        # update error
                        error = Out[out, :] - y[i, out]
                        # update output weights
                        dW_readout = -np.dot(error, (P_readout_old_X/den_readout).T)
                        self._WXOut[out, :] = self._WXOut[out, :] + dW_readout
                        if self.save_history:
                            # store change in weights
                            self._dW_readout_len[i] = np.sqrt(np.dot(dW_readout, dW_readout.T))

            if self.save_history or get_target_innate_X:
                self._X_history[i, :] = Xi.T

            if self.save_history:
                # store output
                self._Out_history[i, :] = Out.T
                self._WXOut_len[i] = np.sqrt(np.sum(np.reshape(self._WXOut**2, (self._num_outputs*self.num_units, 1))))
                self._WXX_len[i] = np.sqrt(np.sum(np.reshape(self._WXX**2, (self.num_units**2, 1))))

            if self.verbose:
                bar.update(1)

        # get target from innate trajectory
        if get_target_innate_X:
            self._Target_innate_X = self._X_history
        elif not hasattr(self, '_Target_innate_X'):
            self._Target_innate_X = []

        return self


    def predict(self, X):
        random_state = check_random_state(self.random_state)
        self._scale = self.g/np.sqrt(self.p_connect*self.num_units)

        X = check_array(X, copy=self.copy_X)
        check_is_fitted(self, '_WXX')

        self._num_inputs = X.shape[1]
        n_steps = X.shape[0]

        noise_amp = self.noise_amplitude

        Out_history = np.zeros((n_steps, self._num_outputs))
        if self.save_history:
            self._X_history = np.zeros((n_steps, self.num_units))
            self._Out_history = Out_history
            # auxiliary variables
            self._WXOut_len = np.zeros((n_steps, 1))
            self._WXX_len = np.zeros((n_steps, 1))

        # initial conditions
        Xv = 2*random_state.rand(self.num_units, 1)-1
        Xi = np.tanh(Xv)
        Out = np.zeros((self._num_outputs, 1))

        if self.verbose:
            bar = tqdm(total = n_steps)

        # integration loop
        for i in range(n_steps):
            # if rem(i,round(n_steps/10)) == 0 && (TRAIN_RECURR == 1 || TRAIN_READOUT == 1)
            #     fprintf('.');
            # end

            Input = X[i,:].T[:, np.newaxis]

            # update units
            noise = noise_amp*random_state.randn(self.num_units, 1)*np.sqrt(self.dt)
            Xv_current = np.dot(self._WXX, Xi) + np.dot(self._WInputX, Input) + noise
            Xv = Xv + ((-Xv + Xv_current)/self.tau)*self.dt
            Xi = np.tanh(Xv)
            Out = np.dot(self._WXOut, Xi)

            Out_history[i, :] = Out.T
            if self.save_history:
                # store output
                self._X_history[i, :] = Xi.T
                self._Out_history = Out_history
                self._WXOut_len[i] = np.sqrt(np.sum(np.reshape(self._WXOut**2, (self._num_outputs*self.num_units, 1))))
                self._WXX_len[i] = np.sqrt(np.sum(np.reshape(self._WXX**2, (self.num_units**2, 1))))

            if self.verbose:
                bar.update(1)

        return Out_history


    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import mean_squared_error

        X, y = check_X_y(X, y, copy=self.copy_X, multi_output=True)
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, sample_weight=sample_weight)