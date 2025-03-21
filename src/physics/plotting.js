import { deltaBarrier, gaussianBarrier, gravityPotential, squareWellPotential } from './potentialHelpers';
import { airyTransformInitialWavefunction, airyTransformClosedForm, probabilityDensity } from '../mathematics/airtransform';
import { airyAi } from '../mathematics/airy';

/**
 * Interpolates velocity at a given position using probability current and density
 */
export const interpolateVelocity = (pos, zArr, currentRow, probRow) => {
  if (pos <= zArr[0]) return currentRow[0] / (probRow[0] || 1e-6);
  if (pos >= zArr[zArr.length - 1]) return currentRow[zArr.length - 1] / (probRow[zArr.length - 1] || 1e-6);
  
  let j = 0;
  while (j < zArr.length - 1 && zArr[j + 1] < pos) {
    j++;
  }
  
  const frac = (pos - zArr[j]) / (zArr[j + 1] - zArr[j]);
  const currentInterp = currentRow[j] + frac * (currentRow[j + 1] - currentRow[j]);
  const probInterp = probRow[j] + frac * (probRow[j + 1] - probRow[j]);
  
  if (Math.abs(probInterp) < 1e-6) return 0;
  return currentInterp / probInterp;
};

/**
 * Generates data for the probability density heatmap
 */
export const generateHeatmapData = (probArr, zArray, tArray) => {
  if (!probArr || probArr.length === 0) return [];
  
  return [{
    z: probArr,
    x: zArray,
    y: tArray,
    type: 'heatmap',
    colorscale: 'Viridis',
    reversescale: false,
    showscale: true,
    zsmooth: 'best',
    zauto: false,
    zmin: 0,
    zmax: 0.5,
    name: '|ψ(z,t)|²'
  }];
};

/**
 * Generates data for the 1D probability density plot with potential
 */
export const generateLineData = (probArr, tIndex, zArray, barrierType, params) => {
  if (!probArr || probArr.length === 0) return [];
  
  const { deltaZ0, deltaAlpha, gaussZ0, gaussV0, gaussSigma, 
          gauss2Z0, gauss2V0, gauss2Sigma, xMin, xMax, Nx } = params;
  
  const slice = probArr[tIndex] || [];
  const potentialValues = zArray.map(z => {
    let barrier;
    if (barrierType === 'delta') {
      barrier = deltaBarrier(z, deltaZ0, deltaAlpha, (xMax - xMin) / Nx);
    } else if (barrierType === 'gaussian') {
      barrier = gaussianBarrier(z, gaussZ0, gaussV0, gaussSigma);
    } else if (barrierType === 'doubleGaussian') {
      barrier = gaussianBarrier(z, gaussZ0, gaussV0, gaussSigma)
              + gaussianBarrier(z, gauss2Z0, gauss2V0, gauss2Sigma);
    } else if (barrierType === 'squareWell') {
      barrier = squareWellPotential(z, gaussZ0, gaussV0, gaussSigma);
    }
    return barrier + gravityPotential(z);
  });

  const indices = zArray.map((z, idx) => ({ z, idx }))
                       .filter(item => item.z >= xMin && item.z <= xMax)
                       .map(item => item.idx);
  
  const plotZ = indices.map(i => zArray[i]);
  const plotSlice = indices.map(i => slice[i]);
  const plotPotential = indices.map(i => potentialValues[i]);

  return [
    {
      x: plotZ,
      y: plotSlice,
      type: 'scatter',
      mode: 'lines',
      name: `|ψ(z, t=${tIndex})|²`,
      line: { color: 'blue' },
    },
    {
      x: plotZ,
      y: plotPotential,
      type: 'scatter',
      mode: 'lines',
      name: 'V(z)',
      line: { color: 'red' },
      yaxis: 'y2',
    },
  ];
};

/**
 * Generates data for the arrival time density plot
 */
export const generateArrivalTimeData = (params) => {
  const { deltaAlpha, detectorL, z0Packet, p0Packet, sigmaPacket, xMin, xMax, Nx } = params;
  const tauMax = 10;
  const numPoints = 400;
  const tauVals = [];
  const piVals = [];

  for (let i = 0; i < numPoints; i++) {
    const tau = tauMax * i / (numPoints - 1);
    tauVals.push(tau);

    // const psiHatAiryBasis = airyTransformInitialWavefunction(
    //   detectorL + tau * tau, 
    //   z0Packet, 
    //   p0Packet, 
    //   sigmaPacket, 
    //   xMin, 
    //   xMax, 
    //   Nx
    // );

	const psiHatAiryBasis = airyTransformClosedForm(
		detectorL + tau * tau,
		z0Packet,
		sigmaPacket
	);

    const psiHatAiryBasisProbability = probabilityDensity(psiHatAiryBasis);
    const numerator = 2 * tau * psiHatAiryBasisProbability;

    const aiValue = airyAi(-detectorL - tau * tau);
    const denominator = 1 + Math.pow(2 * Math.PI * tau * deltaAlpha * Math.pow(aiValue, 2), 2);

    const piVal = tau <= 0 ? 0 : numerator / denominator;
    piVals.push(piVal);
  }

  return [
    { 
      x: tauVals, 
      y: piVals, 
      type: 'scatter', 
      mode: 'lines+markers', 
      name: 'Arrival Time Density (Π)' 
    },
  ];
};

/**
 * Generates data for the trajectory plot with probability current
 */
export const generateTrajectoryData = (params) => {
  const { currentArr, trajectories, zArray, tArray, dt, trajFinesse, 
          showTrajectories, detectorL } = params;
  
  const data = [];
  
  if (currentArr && currentArr.length > 0) {
    data.push({
      z: currentArr,
      x: zArray,
      y: tArray,
      type: 'heatmap',
      colorscale: 'RdBu',
      reversescale: true,
      showscale: true,
      colorbar: { title: 'j' },
      name: 'Probability Current'
    });

    data.push({
      x: [-detectorL, -detectorL],
      y: [Math.min(...tArray), Math.max(...tArray)],
      mode: 'lines',
      line: { dash: 'dot', color: 'red', width: 2 },
      name: 'Detector Position (z=-L)'
    });
  }

  if (showTrajectories && trajectories && trajectories.length > 0) {
    const refinedLength = trajectories[0].length;
    const tRefined = Array.from(
      { length: refinedLength }, 
      (_, j) => j * (dt / trajFinesse)
    );
    
    trajectories.forEach(traj => {
      data.push({
        x: traj,
        y: tRefined,
        mode: 'lines',
        name: '',
        showlegend: false,
        line: { width: 0.5, color: '#333333' },
      });
    });
  }

  return data;
};

/**
 * Generates data for the arrival times distribution plot
 */
export const generateArrivalTimesData = (currentArr, tArray, zArray, detectorL) => {
  if (!currentArr || currentArr.length === 0 || zArray.length === 0) return [];

  let detectorIndex = zArray.findIndex((z) => z >= -detectorL);
  if (detectorIndex === -1) detectorIndex = zArray.length - 1;

  const currentAtDetector = currentArr.map(row => -1 * row[detectorIndex]);

  return [{
    x: tArray,
    y: currentAtDetector,
    type: 'scatter',
    mode: 'lines',
    name: 'Simulated Arrival-Time Distribution (j at -L)',
    line: { color: 'blue' }
  }];
};
