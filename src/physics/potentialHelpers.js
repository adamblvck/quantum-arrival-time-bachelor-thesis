import { hbar, m, g } from './constants';

 /**
 * ---------------------------------------
 * Wavefunction & Potential Helper Functions
 * ---------------------------------------
 */
export const gravityPotential = (z) => m * g * z;

export const deltaBarrier = (z, z0, alpha, dz) => {
   const sigma = dz / 2;
   return alpha * Math.exp( -((z - z0) ** 2) / (2 * sigma ** 2) ) / (Math.sqrt(2 * Math.PI) * sigma);
 };

export const gaussianBarrier = (z, z0, V0, sigma) => {
   return V0 * Math.exp( -((z - z0) ** 2) / (2 * sigma ** 2) );
};

export const squareWellPotential = (z, z0, V0, width) => {

	if (z < z0 - width/2 || z > z0 + width/2) {
		return 0;
	} else {
		return V0;
	}
};

export const initialWavefunction = (z, z0, p0, sigma0 = 1.0) => {

	// Normalization factor - required for correct quantum mechanics
	// Well-known result from [REFERENCE]
	const normFactor = 1.0 / Math.pow(2.0 * Math.PI * sigma0 ** 2, 0.25);

	// p0 - Initial nudge in momentum space
	const phase = (p0 * (z - z0)) / hbar; 

	// Initial wavefunction in position space
	const realPart = normFactor * Math.exp( -((z - z0) ** 2) / (2.0 * sigma0 ** 2) ) * Math.cos(phase);
	const imagPart = normFactor * Math.exp( -((z - z0) ** 2) / (2.0 * sigma0 ** 2) ) * Math.sin(phase);
	return { re: realPart, im: imagPart };
 };




