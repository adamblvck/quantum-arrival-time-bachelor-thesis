import { deltaBarrier, gaussianBarrier, initialWavefunction, gravityPotential, squareWellPotential } from './potentialHelpers';
import { applyPotentialHalfStep, applyKineticFullStep, applyAbsorbingMask, applyCosineMask, applyCAP } from './splitOperatorHelpers';
import { fft, ifft } from 'fft-js';

export const simulateN = ({
  xMin, xMax, Nx, nSteps, dt, hbar, m, barrierType, deltaZ0, deltaAlpha,
  gaussZ0, gaussV0, gaussSigma, gauss2Z0, gauss2V0, gauss2Sigma, useSuperposition,
  z0Packet, p0Packet, sigmaPacket, z0Packet2, p0Packet2, sigmaPacket2
}) => {

  // setup parameters
  const plottingRange = xMax - xMin;
  const simRange = plottingRange * 4;
  const center = (xMin + xMax) / 2;
  const simXMin = center - simRange / 2;
  const simXMax = center + simRange / 2;
  const dz = (simXMax - simXMin) / Nx;
  const zArr = new Array(Nx);
  for (let i = 0; i < Nx; i++) {
    zArr[i] = simXMin + i * dz;
  }

  // Build potential array based on the selected barrierType
  let barrierArr = new Array(Nx).fill(0);
  if (barrierType === 'delta') {
    for (let i = 0; i < Nx; i++) {
      barrierArr[i] = deltaBarrier(zArr[i], deltaZ0, deltaAlpha, dz);
    }
  } else if (barrierType === 'gaussian') {
    for (let i = 0; i < Nx; i++) {
      barrierArr[i] = gaussianBarrier(zArr[i], gaussZ0, gaussV0, gaussSigma);
    }
  } else if (barrierType === 'doubleGaussian') {
    for (let i = 0; i < Nx; i++) {
      barrierArr[i] = gaussianBarrier(zArr[i], gaussZ0, gaussV0, gaussSigma)
                    + gaussianBarrier(zArr[i], gauss2Z0, gauss2V0, gauss2Sigma);
    }
  } else if (barrierType == "squareWell") {
    for (let i = 0; i < Nx; i++) {
      barrierArr[i] = squareWellPotential(zArr[i], gaussZ0, gaussV0, gaussSigma);
    }
  }
  // Add the gravitational potential to each grid point.
  for (let i = 0; i < Nx; i++) {
    barrierArr[i] += gravityPotential(zArr[i]);
  }

  // Build k-array for momentum space
  const kArr = new Array(Nx);
  for (let i = 0; i < Nx; i++) {
    const kIndex = (i < Nx/2) ? i : i - Nx;
    kArr[i] = (2.0 * Math.PI * kIndex) / (Nx * dz);
  }

  // Build kinetic factors for momentum space step
  const kineticRe = new Array(Nx);
  const kineticIm = new Array(Nx);
  for (let i = 0; i < Nx; i++) {
    const phase = -(kArr[i] * kArr[i] * dt) / (2 * m);
    kineticRe[i] = Math.cos(phase);
    kineticIm[i] = Math.sin(phase);
  }

  // Initialize wavefunction
  const psiRe = new Array(Nx);
  const psiIm = new Array(Nx);
  for (let i = 0; i < Nx; i++) {
    if (useSuperposition) {
      // Compute two wavefunctions and sum them:
      const wf1 = initialWavefunction(zArr[i], z0Packet, p0Packet, sigmaPacket);
      const wf2 = initialWavefunction(zArr[i], z0Packet2, p0Packet2, sigmaPacket2);
      psiRe[i] = wf1.re + wf2.re;
      psiIm[i] = wf1.im + wf2.im;
    } else {
      const { re, im } = initialWavefunction(zArr[i], z0Packet, p0Packet, 1.0);
      psiRe[i] = re;
      psiIm[i] = im;
    }
  }

  // Prepare arrays for probability density and current at each time step
  const probStorage = [];
  probStorage.length = nSteps;
  const currentStorage = [];
  currentStorage.length = nSteps;

  const dtOver2hbar = dt / (2 * hbar);

  // Initialize separate arrays for left and right losses
  const lossArrLeft = new Array(nSteps);
  const lossArrRight = new Array(nSteps);
  const cumulativeLossArrLeft = new Array(nSteps);
  const cumulativeLossArrRight = new Array(nSteps);
  let runningLossLeft = 0;
  let runningLossRight = 0;

  // Main time evolution loop
  for (let step = 0; step < nSteps; step++) {
    // Store probability density at this step
    const probRow = new Array(Nx);
    for (let i = 0; i < Nx; i++) {
      probRow[i] = psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
    }
    probStorage[step] = probRow;

    // Compute probability current using central finite differences
    const currentRow = new Array(Nx);
    for (let i = 1; i < Nx - 1; i++) {
      const dRe = (psiRe[i + 1] - psiRe[i - 1]) / (2 * dz);
      const dIm = (psiIm[i + 1] - psiIm[i - 1]) / (2 * dz);
      currentRow[i] = (hbar / m) * (psiRe[i] * dIm - psiIm[i] * dRe);
    }
    currentRow[0] = currentRow[1];
    currentRow[Nx - 1] = currentRow[Nx - 2];
    currentStorage[step] = currentRow;

    // First half-step in potential
    applyPotentialHalfStep(psiRe, psiIm, barrierArr, dtOver2hbar);

    // Full kinetic step:
    const inputRe = new Float64Array(Nx);
    const inputIm = new Float64Array(Nx);
    const outputRe = new Float64Array(Nx);
    const outputIm = new Float64Array(Nx);
    for (let i = 0; i < Nx; i++) {
      inputRe[i] = psiRe[i];
      inputIm[i] = psiIm[i];
    }
    const signal = Array.from(inputRe).map((re, i) => [re, inputIm[i]]);

    // fourier transform into momentum space
    const transformed = fft(signal);
    for (let i = 0; i < Nx; i++) {
      outputRe[i] = transformed[i][0];
      outputIm[i] = transformed[i][1];
    }

    // apply kinetic step in momentum space
    applyKineticFullStep(outputRe, outputIm, kineticRe, kineticIm);

    // inverse fourier transform back into position space
    const signalInv = Array.from(outputRe).map((re, i) => [re, outputIm[i]]);
    const transformedInv = ifft(signalInv);
    for (let i = 0; i < Nx; i++) {
      psiRe[i] = transformedInv[i][0];
      psiIm[i] = transformedInv[i][1];
    }

    // Second half-step in potential
    applyPotentialHalfStep(psiRe, psiIm, barrierArr, dtOver2hbar);

    // Apply absorbing mask and capture separate losses
    // const { stepLossLeft, stepLossRight } = applyAbsorbingMask(psiRe, psiIm, zArr, { simXMin, simXMax });
    const { capLossLeft: stepLossLeft, capLossRight: stepLossRight } = applyCAP(psiRe, psiIm, zArr, { xMin, xMax, dt, hbar, W0: 1.0 });
    // const { stepLossLeft, stepLossRight } = applyCosineMask(psiRe, psiIm, zArr, { xMin, xMax });
    
    lossArrLeft[step] = stepLossLeft;
    lossArrRight[step] = stepLossRight;
    runningLossLeft += stepLossLeft;
    runningLossRight += stepLossRight;
    cumulativeLossArrLeft[step] = runningLossLeft;
    cumulativeLossArrRight[step] = runningLossRight;
  }

  // Build time array
  const tArr = new Array(nSteps);
  for (let i = 0; i < nSteps; i++) {
    tArr[i] = i * dt;
  }

  return { zArr, tArr, probArr: probStorage, currentArr: currentStorage, lossArrLeft, lossArrRight, cumulativeLossArrLeft, cumulativeLossArrRight };
};
