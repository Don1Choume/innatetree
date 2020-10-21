import numpy as np

class DiscreteLorenz(object):
    def __init__(self, a=10, r=28, b=8/3, dt=0.001):
        self.a = a
        self.r = r
        self.b = b
        self.dt = dt

    def _calc_stepwise(self, x: float, y: float, z: float):
        x_ = x + self.a*(y - x)*self.dt
        y_ = y + (-x*z + self.r*x - y)*self.dt
        z_ = z + (x*y - self.b*z)*self.dt
        return x_, y_, z_

    def get_series(self, n_step: int, initial_state: tuple, add_noise_step=[]):
        """
        Lorenz orbit generator.
        Args:
            n_step (int): step size of Lorenz series
            initial_state (tuple): initial state of Lorenz system. For example:
                                    (x, y, z) =  (0.02, 0.01, 0.03)
                                    (x, y, z) =  (-1, 0, 1) 
            add_noise_step (list, optional): noise description. Defaults to [].
                                    list of (step, (noise_x, noise_y, noise_z))

        Returns:
            numpy.ndarray(float): step x [xyz]
        """
        n_step = int(n_step)
        series = np.zeros((n_step, 3))
        noise = np.zeros((n_step, 3))

        [noise.put(np.arange(3*int(s), 3*(int(s)+1)), np.array(n))
            for s, n in add_noise_step]

        series[0, :] = initial_state

        # vectorizeの方がコスト高い
        [series.put(np.arange(3*s, 3*(s+1)), self._calc_stepwise(*(series[s-1, :]+noise[s-1, :])))
            for s in range(1, n_step)]

        return series
        


if __name__=='__main__':
    import sys
    from pathlib import Path
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.append(str(src_dir))