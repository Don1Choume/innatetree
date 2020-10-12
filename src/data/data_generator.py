from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy import io

class BaseGenData(object):
    def __init__(self):
        raise NotImplementedError()

    def gen_input(self):
        raise NotImplementedError()

    def gen_output(self):
        raise NotImplementedError()

    def gen_train_window(self):
        pass


class MatlabGen(BaseGenData):
    def __init__(self, target='timing'):
        self.target = target
        prj_dir = Path(__file__).resolve().parents[2]
        ref_dir = prj_dir/'references'/'original'
        tim_mat = ref_dir/'DAC_timing_training'/'DAC_timing_recurr800_p0.1_g1.5.mat'
        hnd_mat = ref_dir/'DAC_handwriting_training'/'DAC_handwriting_recurr800_p0.1_g1.5.mat'
        if target == 'timing':
            self.mat = io.loadmat(str(tim_mat))
        else:
            self.mat = io.loadmat(str(hnd_mat))
        self.dt = self.mat['dt']
        self.tmax = self.mat['tmax']

    def gen_input(self):
        if self.target == 'timing':
            input_pattern = np.squeeze(self.mat['input_pattern'])
        else:
            input_pattern = np.squeeze(np.concatenate(self.mat['input_pattern'], axis=1).T)
        return input_pattern

    def gen_output(self):
        if self.target == 'timing':
            target_Out = np.squeeze(self.mat['target_Out'])
        else:
            target_Out = np.squeeze(np.concatenate(self.mat['target_Out'], axis=1).T)
        return target_Out

    def gen_train_window(self):
        return (np.arange(int(self.tmax/self.dt)) >= int(self.mat['start_train']/self.dt)) &\
                np.arange(int(self.tmax/self.dt)) < int(self.mat['end_train']/self.dt)


class TimingGen(BaseGenData):
    def __init__(self,
                 dt=1,
                 interval=1000,
                 start_pulse=200,
                 reset_duration=50,
                 end_interval=150,
                 end_duration=200,
                 input_pulse_value=5.0):
        self.dt = dt
        self.interval = interval
        self.start_pulse = start_pulse
        self.reset_duration = reset_duration
        self.end_interval = end_interval
        self.end_duration = end_duration
        self.input_pulse_value = input_pulse_value

        self.tmax = self.interval + \
                    self.start_pulse + \
                    self.reset_duration + \
                    self.end_interval + \
                    self.end_duration
        self.n_steps = np.round(self.tmax/self.dt).astype(int)

    def gen_input(self, num_inputs=1):
        self.num_inputs = num_inputs
        self.start_train = self.start_pulse + self.reset_duration
        self.end_train = self.start_train + self.interval + self.end_interval
        ####
        start_pulse_n = np.round(self.start_pulse/self.dt).astype(int)
        reset_duration_n = np.round(self.reset_duration/self.dt).astype(int)
        start_train_n = np.round(self.start_train/self.dt).astype(int)
        end_train_n = np.round(self.end_train/self.dt).astype(int)

        input_pattern = np.zeros((self.n_steps, self.num_inputs))
        input_pattern[start_pulse_n:(start_pulse_n+reset_duration_n), 0] = \
            self.input_pulse_value*np.ones(reset_duration_n)
        return input_pattern

    def gen_output(self, ready_level=0.2, peak_level=1, peak_width=30):
        peak_time = self.start_train + self.interval
        time_axis = np.arange(0, self.tmax, self.dt)
        bell = norm.pdf(time_axis, loc=peak_time, scale=peak_width)
        bell_max = np.max(bell)
        target_Out = ready_level + ((peak_level-ready_level)/bell_max)*bell
        return target_Out[:, np.newaxis]

    def gen_train_window(self):
        start_train = self.start_pulse + self.reset_duration
        end_train = start_train + self.interval + self.end_interval
        return (np.arange(self.n_steps) >= int(start_train/self.dt)) &\
                (np.arange(self.n_steps) < int(end_train/self.dt))


class HandWritingGen(object):
    def __init__(self,
                 dt=1,
                 start_pulse=200,
                 reset_duration=50,
                 interval_1=1322,
                 interval_2=1234,
                 input_pulse_value=2.0):
        self.dt = dt
        self.start_pulse = start_pulse
        self.reset_duration = reset_duration
        self.interval_1 = interval_1
        self.interval_2 = interval_2
        self.start_train = start_pulse + reset_duration
        self.end_train_1 = self.start_train + interval_1
        self.end_train_2 = self.start_train + interval_2
        self.input_pulse_value = input_pulse_value

        self.tmax = np.max([self.end_train_1, self.end_train_2]) + 1000
        self.n_steps = np.round(self.tmax/self.dt).astype(int)

        prj_dir = Path(__file__).resolve().parents[2]
        ref_dir = prj_dir/'references'/'original'
        hnd_mat = ref_dir/'DAC_handwriting_training'/'DAC_handwriting_output_targets.mat'
        self.mat = io.loadmat(str(hnd_mat))

    def gen_input(self, num_inputs=2):
        self.num_inputs = num_inputs
        start_pulse_n = np.round(self.start_pulse/self.dt).astype(int)
        reset_duration_n = np.round(self.reset_duration/self.dt).astype(int)
        interval_1_n = np.round(self.interval_1/self.dt).astype(int)
        interval_2_n = np.round(self.interval_2/self.dt).astype(int)

        input_pattern = np.zeros((2*self.n_steps, self.num_inputs))
        input_pattern[start_pulse_n:(start_pulse_n+reset_duration_n), 0] = \
            self.input_pulse_value*np.ones(reset_duration_n)
        input_pattern[(self.n_steps+start_pulse_n):(self.n_steps+start_pulse_n+reset_duration_n), 1] = \
            self.input_pulse_value*np.ones(reset_duration_n)
        return input_pattern

    def gen_output(self, num_outputs=2):
        self.num_outputs = num_outputs
        start_train_n = np.round(self.start_train/self.dt).astype(int)
        end_train_1_n = np.round(self.end_train_1/self.dt).astype(int)
        end_train_2_n = np.round(self.end_train_2/self.dt).astype(int)
        target_Out = np.zeros((2*self.n_steps, self.num_outputs))
        target_Out[start_train_n:end_train_1_n, :] = self.mat['chaos'].T
        target_Out[(self.n_steps+start_train_n):(self.n_steps+end_train_2_n), :] = self.mat['neuron'].T
        return target_Out

    def gen_train_window(self):
        start_train1 = self.start_train
        end_train1 = self.end_train_1
        start_train2 = self.start_train + self.n_steps
        end_train2 = self.end_train_2 + self.n_steps
        return ((np.arange(2*self.n_steps) >= int(start_train1/self.dt)) &\
                (np.arange(2*self.n_steps) < int(end_train1/self.dt))) |\
                ((np.arange(2*self.n_steps) >= int(start_train2/self.dt)) &\
                (np.arange(2*self.n_steps) < int(end_train2/self.dt)))


if __name__=="__main__":
    data_gen1 = MatlabGen()
    # data_gen1 = MatlabGen('hand')
    data_gen2 = TimingGen()
    # data_gen2 = HandWritingGen()
    inputs1 = data_gen1.gen_input()
    outputs1 = data_gen1.gen_output()
    print(inputs1.shape)
    print(outputs1.shape)
    inputs2 = data_gen2.gen_input()
    outputs2 = data_gen2.gen_output()
    print(inputs2.shape)
    print(outputs2.shape)