// Airy Transform of a wavefunction - the representation of a wavefunction in the Airy Basis


import { initialWavefunction } from '../physics/potentialHelpers';
import { airyAi } from './airy';

export function probabilityDensity(psi) {
	// assume that psi is a complex number, with re and im fields
	return psi.re ** 2 + psi.im ** 2;
}

/**
 * Numerically computes
 *     ψ̂(E) = ∫ Ai[z - E] * ψ(z, 0) dz
 * in the dimensionless form used by the paper for v₀ = 0.
 */
export function airyTransformInitialWavefunction(
    E,      // dimensionless "energy" variable in the paper
    z0,     // where the Gaussian is centered
    sigma,  // width parameter from eq. (A1)
    zMin = -20,
    zMax =  20,
    numPoints = 2000
  ) {
    const dz = (zMax - zMin) / numPoints;
    let sumRe = 0;
    let sumIm = 0;
  
    for (let i = 0; i < numPoints; i++) {
      const z = zMin + i * dz;
      // Paper's Gaussian at z
      const psiVal = initialWavefunction(z, z0, sigma);
  
      // Airy-Ai function at (z - E)
      const AiVal = airyAi(z - E);
  
      // Multiply and accumulate
      sumRe += AiVal * psiVal.re;
      sumIm += AiVal * psiVal.im;
    }
    // Factor out the width of each slice
    sumRe *= dz;
    sumIm *= dz;
  
    return { re: sumRe, im: sumIm };
  }

  export function airyTransformClosedForm(E, z0, sigma) {
    // Prefactor: (2σ√π)^{1/2}
    const prefactor = Math.sqrt(2 * sigma * Math.sqrt(Math.PI));
  
    // Exponential argument: [ σ²/2 * (σ⁴/6 + z0 - E) ]
    const exponentArg = 0.5 * (sigma**2 /2 ) * ((sigma**4) / 6 + z0 - E);
  
    // Airy argument:  (σ⁴)/4 + z0 - E
    const airyArg = (sigma**4)/4 + z0 - E;
  
    return {
        re: prefactor *
            Math.exp(exponentArg) *
            airyAi(airyArg),
        im: 0
    };
  }