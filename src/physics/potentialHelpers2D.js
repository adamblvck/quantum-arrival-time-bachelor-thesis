// 2D Potential and Initial Wavefunction Helpers

import { hbar, m, g } from './constants';

// Gravity acting in the y-direction:
export const gravityPotential = (y) => m * g * y;

// A delta-barrier in x (implemented as a narrow Gaussian):
export const deltaBarrier = (x, x0, alpha, dx) => {
  const sigma = dx / 2;
  return alpha * Math.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) / (Math.sqrt(2 * Math.PI) * sigma);
};

// 2D initial wavefunction: a Gaussian wave packet centered at (x0, y0) with momentum (p0x, p0y).
export const initialWavefunction2D = (x, y, x0, y0, p0x, p0y, sigma0 = 1.0) => {
  const normFactor = 1.0 / Math.pow(2 * Math.PI * sigma0 ** 2, 0.5);
  const phase = (p0x * (x - x0) + p0y * (y - y0)) / hbar;
  const gaussian = Math.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma0 ** 2)));
  return {
    re: normFactor * gaussian * Math.cos(phase),
    im: normFactor * gaussian * Math.sin(phase)
  };
};
