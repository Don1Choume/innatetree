import numpy as np
from matplotlib import pyplot as plt


def visualize_IO(inputs, outputs, dt=1, savename=None):
    x_axis = (np.arange(len(inputs))*dt).squeeze()

    if len(inputs.shape) < 3:
        inputs = inputs[np.newaxis, :, :]
        outputs = outputs[np.newaxis, :, :]

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(121)
    for idx_1 in range(inputs.shape[1]):
        ax1.plot(x_axis, inputs[:, idx_1], label='idx = {0}'.format(idx_1))
    ax1.set_title('Input')
    ax2 = fig.add_subplot(122)
    for idx_1 in range(outputs.shape[1]):
        ax2.plot(x_axis, outputs[:, idx_1], label='idx = {0}'.format(idx_1))
    ax2.set_title('Output')
    # ax1.grid()
    # ax1.legend()
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()
    plt.close()


if __name__=="__main__":
    import sys
    from pathlib import Path
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.append(str(src_dir))

    from data import MatlabGen, TimingGen, HandWritingGen

    # data_gen = MatlabGen()
    data_gen = MatlabGen('hand')
    # data_gen = TimingGen()
    # data_gen = HandWritingGen()
    inputs = data_gen.gen_input()
    outputs = data_gen.gen_output()
    visualize_IO(inputs, outputs, dt=data_gen.dt)