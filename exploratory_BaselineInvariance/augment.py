# augment.py
import numpy as np
import math

class BaselineGenerator:
    """
    Augment spectra with additive baseline, fringe, scaling, and noise.
    """

    def __init__(self, x, poly_scale=(0.0, 0.002), poly_order=3,
                 fringe_prob=0.4, fringe_amp=(0.0, 0.004), fringe_freq=(0.0005, 0.003),
                 vp_scale=(0.9, 1.1), noise_std=(0.0, 0.003)):
        self.x = x
        self.N = len(x)
        self.poly_scale = poly_scale
        self.poly_order = poly_order
        self.fringe_prob = fringe_prob
        self.fringe_amp = fringe_amp
        self.fringe_freq = fringe_freq
        self.vp_scale = vp_scale
        self.noise_std = noise_std

        # normalize x for stable polynomial basis
        self.xn = (x - x.mean()) / (x.std() + 1e-12)

    def sample_polynomial(self, B):
        coeffs = np.random.uniform(self.poly_scale[0], self.poly_scale[1],
                                   size=(B, self.poly_order+1))
        for k in range(coeffs.shape[1]):
            coeffs[:,k] *= (0.5**k)
        out = np.zeros((B, self.N), dtype=float)
        for k in range(coeffs.shape[1]):
            out += np.outer(coeffs[:,k], self.xn**k)
        return out

    def sample_fringe(self, B):
        amps = np.random.uniform(self.fringe_amp[0], self.fringe_amp[1], size=(B,1))
        freqs = np.random.uniform(self.fringe_freq[0], self.fringe_freq[1], size=(B,1))
        phases = np.random.uniform(0, 2*np.pi, size=(B,1))
        xv = self.x[None,:]
        return amps * np.sin(2*np.pi * freqs * xv + phases)

    def sample(self, batch_y):
        B = batch_y.shape[0]
        baseline = self.sample_polynomial(B)
        if np.random.rand() < self.fringe_prob:
            baseline += self.sample_fringe(B)
        scale = np.random.uniform(self.vp_scale[0], self.vp_scale[1], size=(B,1))
        y = batch_y * scale + baseline
        noise = np.random.normal(0, np.random.uniform(self.noise_std[0], self.noise_std[1], size=(B,1)), size=(B,self.N))
        y = y + noise
        return y.astype(np.float32), baseline.astype(np.float32), scale.astype(np.float32)
