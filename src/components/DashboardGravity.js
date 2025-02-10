import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import { fft, ifft } from 'fft-js';
import { debounce } from 'lodash';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { explanations, InfoButton } from './explanations';

/**
 * Example InfoModal component
 */
const InfoModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const markdown = `
# Quantum Simulation Info

This dashboard simulates the time evolution of a wavefunction $\\psi(z,t)$ under a 1D potential 
using the split-operator method. The Hamiltonian of the system is:

$$
\\hat{H} = -\\frac{\\hbar^2}{2m}\\frac{\\partial^2}{\\partial z^2} + V(z) + mgz
$$

The probability density is given by $|\\psi(z,t)|^2$, and the probability current $j(z,t)$ is:

$$
j(z,t) = \\frac{\\hbar}{m}\\text{Im}\\left(\\psi^*\\frac{\\partial\\psi}{\\partial z}\\right)
$$

For Bohmian trajectories, particle velocities are computed using $v(z,t) = j(z,t)/|\\psi(z,t)|^2$.
The initial wavefunction is a Gaussian wave packet:

$$
\\psi(z,0) = \\frac{1}{(2\\pi\\sigma_0^2)^{1/4}}\\exp\\left(-\\frac{(z-z_0)^2}{4\\sigma_0^2} + \\frac{ip_0z}{\\hbar}\\right)
$$

When using superposition, we add two such wave packets with different parameters.
  `;

  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded shadow-md max-w-2xl w-full overflow-y-auto max-h-[90vh]">
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            className="space-y-4 text-gray-700"
          >
            {markdown}
          </ReactMarkdown>
        </div>
        <button 
          onClick={onClose} 
          className="mt-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
        >
          Close
        </button>
      </div>
    </div>
  );
};

const ExplanationModal = ({ isOpen, onClose, content }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded shadow-md max-w-2xl w-full overflow-y-auto max-h-[90vh]">
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            className="space-y-4 text-gray-700"
          >
            {content}
          </ReactMarkdown>
        </div>
        <button 
          onClick={onClose} 
          className="mt-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
        >
          Close
        </button>
      </div>
    </div>
  );
};

const DB_NAME = 'quantum-sim-db';
const STORE_NAME = 'simulations';

// Initialize IndexedDB
const initDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'timestamp' });
      }
    };
  });
};

const Dashboard = () => {
  /**
   * -------------------------------
   * 1) State for PDE / Potential
   * -------------------------------
   */
  // Available potential types: 'delta', 'gaussian', 'doubleGaussian'
  const [barrierType, setBarrierType] = useState('delta'); 
  
  // Delta-barrier parameters
  const [deltaAlpha, setDeltaAlpha] = useState(15.0);
  const [deltaZ0, setDeltaZ0] = useState(0.0);

  // Gaussian-barrier parameters (for single Gaussian)
  const [gaussV0, setGaussV0] = useState(5.0);
  const [gaussSigma, setGaussSigma] = useState(0.5);
  const [gaussZ0, setGaussZ0] = useState(0.0);

  // NEW: Parameters for the second Gaussian (for doubleGaussian potential)
  const [gauss2V0, setGauss2V0] = useState(5.0);
  const [gauss2Sigma, setGauss2Sigma] = useState(0.5);
  const [gauss2Z0, setGauss2Z0] = useState(5.0);

  // Wave packet initial conditions for the first wavefunction
  const [z0Packet, setZ0Packet] = useState(10.0);
  const [p0Packet, setP0Packet] = useState(0.0);

  // NEW: Additional state for a second wavefunction (for superposition)
  const [useSuperposition, setUseSuperposition] = useState(false);
  const [z0Packet2, setZ0Packet2] = useState(5.0);
  const [p0Packet2, setP0Packet2] = useState(0.0);
  const [sigmaPacket2, setSigmaPacket2] = useState(1.0);

  // PDE solver parameters (xMin/xMax are the plotting limits)
  const [xMin, setXMin] = useState(-20);
  const [xMax, setXMax] = useState(20);
  const [Nx, setNx] = useState(512);
  const [nSteps, setNSteps] = useState(1000);
  const [dt, setDt] = useState(0.01);

  // Detector parameter L (the detector is at z = –L, L > 0)
  const [detectorL, setDetectorL] = useState(1.0);

  // Gravity constants (hard-coded for now)
  const hbar = 1.0;
  const m = 1.0;
  const g = 1.0;

  // New state for storing simulation results:
  const [zArray, setZArray] = useState([]);       // simulation grid
  const [tArray, setTArray] = useState([]);         // simulation time array
  const [probArr, setProbArr] = useState([]);       // probability density [nSteps x Nx]
  const [currentArr, setCurrentArr] = useState([]); // probability current [nSteps x Nx]

  // The currently selected time index (for the slice plot)
  const [tIndex, setTIndex] = useState(0);

  // For playing/pausing the time slider:
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);

  // For localStorage / naming the simulation:
  const [simName, setSimName] = useState('My Simulation');
  const [savedSims, setSavedSims] = useState([]);

  // New state for particle trajectory parameters and computed trajectories:
  const [particleSpawnCenter, setParticleSpawnCenter] = useState(10.0);  // renamed from particleStart
  const [particleSpawnWidth, setParticleSpawnWidth] = useState(1.0);    // renamed from particleEnd
  const [numParticles, setNumParticles] = useState(10);       // number of trajectories
  const [trajFinesse, setTrajFinesse] = useState(1);          // number of substeps per dt for smoother trajectories
  // NEW: Trajectory Integration Factor to control the effective integration dt
  const [trajIntegrationFactor, setTrajIntegrationFactor] = useState(1.0);
  const [showTrajectories, setShowTrajectories] = useState(true); // toggle to show/hide trajectories
  const [trajectories, setTrajectories] = useState([]);       // computed trajectories

  // Info Modal
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Add new state for explanation modals
  const [activeExplanation, setActiveExplanation] = useState(null);

  /**
   * ---------------------------------------
   * 2) Potential Helper Functions (in JS)
   * ---------------------------------------
   */
  const gravityPotential = (z) => m * g * z;

  const deltaBarrier = (z, z0, alpha, dz) => {
    const sigma = dz / 2;
    return alpha * Math.exp( -((z - z0) ** 2) / (2 * sigma ** 2) ) / (Math.sqrt(2 * Math.PI) * sigma);
  };

  const gaussianBarrier = (z, z0, V0, sigma) => {
    return V0 * Math.exp( -((z - z0) ** 2) / (2 * sigma ** 2) );
  };

  const initialWavefunction = (z, z0, p0, sigma0 = 1.0) => {
    const normFactor = 1.0 / Math.pow(2.0 * Math.PI * sigma0 ** 2, 0.25);
    const phase = (p0 * (z - z0)) / hbar; 
    const realPart = normFactor * Math.exp( -((z - z0) ** 2) / (4.0 * sigma0 ** 2) ) * Math.cos(phase);
    const imagPart = normFactor * Math.exp( -((z - z0) ** 2) / (4.0 * sigma0 ** 2) ) * Math.sin(phase);
    return { re: realPart, im: imagPart };
  };

  /**
   * --------------------------------------
   * 3) Split-Operator Step in JavaScript
   * --------------------------------------
   */
  const applyPotentialHalfStep = (psiRe, psiIm, V, dtOver2hbar) => {
    for (let i = 0; i < psiRe.length; i++) {
      const phase = -V[i] * dtOver2hbar;
      const c = Math.cos(phase);
      const s = Math.sin(phase);
      const reOld = psiRe[i];
      const imOld = psiIm[i];
      psiRe[i] = reOld * c - imOld * s;
      psiIm[i] = reOld * s + imOld * c;
    }
  };

  const applyKineticFullStep = (psiRe, psiIm, kineticRe, kineticIm) => {
    for (let i = 0; i < psiRe.length; i++) {
      const reOld = psiRe[i];
      const imOld = psiIm[i];
      const kr = kineticRe[i];
      const ki = kineticIm[i];
      psiRe[i] = reOld * kr - imOld * ki;
      psiIm[i] = reOld * ki + imOld * kr;
    }
  };

  // Apply an absorbing (damping) mask near the edges (with margin = 0.1*(xMax-xMin))
  const applyAbsorbingMask = (psiRe, psiIm, zArr) => {
    const margin = 0.2 * (xMax - xMin);
    const sim_bounce = (xMax - xMin)/2;
    for (let i = 0; i < psiRe.length; i++) {
      let factor = 1;
      if (zArr[i] < xMin - margin) {
        factor = Math.exp(- Math.pow((xMin - margin - zArr[i]) / margin, 2));
      } else if (zArr[i] > xMax + margin) {
        factor = Math.exp(- Math.pow((zArr[i] - (xMax + margin)) / margin, 2));
      }
      psiRe[i] *= factor;
      psiIm[i] *= factor;
    }
  };

  /**
   * ------------------------------------------
   * 4) The "simulateN" function in JavaScript
   * ------------------------------------------
   *
   * In addition to storing |ψ|² (probArr), we also compute a finite-difference approximation
   * for the probability current j and store it in currentArr.
   */
  const simulateN = () => {
    const plottingRange = xMax - xMin;
    const simRange = plottingRange * 2;
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
        const wf1 = initialWavefunction(zArr[i], z0Packet, p0Packet, 1.0);
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
      const transformed = fft(signal);
      for (let i = 0; i < Nx; i++) {
        outputRe[i] = transformed[i][0];
        outputIm[i] = transformed[i][1];
      }
      applyKineticFullStep(outputRe, outputIm, kineticRe, kineticIm);
      const signalInv = Array.from(outputRe).map((re, i) => [re, outputIm[i]]);
      const transformedInv = ifft(signalInv);
      for (let i = 0; i < Nx; i++) {
        psiRe[i] = transformedInv[i][0];
        psiIm[i] = transformedInv[i][1];
      }

      // Second half-step in potential
      applyPotentialHalfStep(psiRe, psiIm, barrierArr, dtOver2hbar);

      // Apply absorbing mask (damping at edges)
      applyAbsorbingMask(psiRe, psiIm, zArr);
    }

    // Build time array
    const tArr = new Array(nSteps);
    for (let i = 0; i < nSteps; i++) {
      tArr[i] = i * dt;
    }

    return { zArr, tArr, probArr: probStorage, currentArr: currentStorage };
  };

  /**
   * ---------------------------------------
   * 5) Helper Functions for Particle Trajectories
   * ---------------------------------------
   *
   * We use a simple Euler integration for trajectories using v(z,t) = j(z,t)/|ψ(z,t)|².
   * The parameter subSteps (trajectory finesse) subdivides each dt.
   * 
   * NEW: The effective time step used in the integration is scaled by trajIntegrationFactor.
   */
  const computeTrajectories = (probArr, currentArr, zArr, tArr, particleCenter, particleWidth, numParticles, subSteps = 1, trajIntegrationFactor = 1.0) => {
    const trajs = [];
    const initialPositions = [];
    for (let i = 0; i < numParticles; i++){
      initialPositions.push(particleCenter - particleWidth/2 + i * (particleWidth) / (numParticles - 1));
    }
    // For each particle, integrate with Euler method using dtLocal = (dt * trajIntegrationFactor)/subSteps
    for (let p = 0; p < numParticles; p++){
      let pos = initialPositions[p];
      const traj = [pos];
      // For each simulation time step...
      for (let t = 0; t < tArr.length - 1; t++){
        const dtLocal = (dt * trajIntegrationFactor) / subSteps;
        // Subdivide each dt
        for (let s = 0; s < subSteps; s++){
          const v = interpolateVelocity(pos, zArr, currentArr[t], probArr[t]);
          pos = pos + v * dtLocal;
          traj.push(pos);
        }
      }
      trajs.push(traj);
    }
    return trajs;
  };

  /**
   * -------------------------------------------
   * 6) Handling the "Run Simulation" button
   * -------------------------------------------
   */
  const handleRunSimulation = () => {
    const { zArr, tArr, probArr, currentArr } = simulateN();
    setZArray(zArr);
    setTArray(tArr);
    setProbArr(probArr);
    setCurrentArr(currentArr);
    setTIndex(0);
    // Compute particle trajectories using the current simulation results and the selected finesse.
    const trajs = computeTrajectories(probArr, currentArr, zArr, tArr, particleSpawnCenter, particleSpawnWidth, numParticles, trajFinesse, trajIntegrationFactor);
    setTrajectories(trajs);
  };

  /**
   * -------------------------------------------
   * 7) Plotly Data for the Different Plots
   * -------------------------------------------
   */
  // Heatmap: |ψ(z,t)|²
  const heatmapData = React.useMemo(() => {
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
  }, [probArr, zArray, tArray]);

  // Slice plot (at the selected tIndex)
  const lineData = React.useMemo(() => {
    if (!probArr || probArr.length === 0) return [];
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
  }, [probArr, tIndex, zArray, barrierType, deltaZ0, deltaAlpha, gaussZ0, gaussV0, gaussSigma, gauss2Z0, gauss2V0, gauss2Sigma, xMin, xMax, Nx]);

  // Define interpolateVelocity first
const interpolateVelocity = (pos, zArr, currentRow, probRow) => {
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

  // Arrival Time Density & Probability Current (analytic) plot
  const arrivalTimeData = React.useMemo(() => {
    const tauMax = 3;
    const numPoints = 400;
    const tauVals = [];
    const piVals = [];
    const jVals = [];
    const gammaVal = deltaAlpha; // using delta barrier strength as gamma
    const LVal = detectorL;
    for (let i = 0; i < numPoints; i++) {
      const tau = tauMax * i / (numPoints - 1);
      tauVals.push(tau);
      const arg = -(LVal + tau * tau);
      // Asymptotic approximations for Airy functions:
      const airyAi = (x) => {
        if (x > 0) {
          return 0.5 / Math.sqrt(Math.PI) * Math.pow(x, -0.25) * Math.exp(- (2/3) * Math.pow(x, 3/2));
        } else {
          const absx = Math.abs(x);
          return 1/Math.sqrt(Math.PI) * Math.pow(absx, -0.25) * Math.sin((2/3)*Math.pow(absx, 3/2) + Math.PI/4);
        }
      };
      const airyAiPrime = (x) => {
        if (x > 0) {
          return -0.5 / Math.sqrt(Math.PI) * Math.pow(x, 0.25) * Math.exp(- (2/3)*Math.pow(x, 3/2));
        } else {
          const absx = Math.abs(x);
          return -1/Math.sqrt(Math.PI) * Math.pow(absx, 0.25) * Math.cos((2/3)*Math.pow(absx, 3/2) + Math.PI/4);
        }
      };
      const AiVal = airyAi(arg);
      const AiPrimeVal = airyAiPrime(arg);
      const numerator = 2 * tau * Math.pow(AiVal, 2);
      const denominator = 1 + Math.pow(2 * tau * gammaVal * AiPrimeVal, 2);
      const piVal = tau <= 0 ? 0 : numerator / denominator;
      piVals.push(piVal);
      jVals.push(piVal); // here j is taken equal to Π for demonstration
    }
    return [
      { x: tauVals, y: piVals, type: 'scatter', mode: 'lines+markers', name: 'Arrival Time Density (Π)' },
      { x: tauVals, y: jVals, type: 'scatter', mode: 'lines+markers', name: 'Probability Current (j)' },
    ];
  }, [deltaAlpha, detectorL]);

  // Particle Trajectories over the Probability Current heatmap
  const trajectoryData = React.useMemo(() => {
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
      
      // Add vertical line at detector position
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
      const tRefined = Array.from({ length: refinedLength }, (_, j) => j * (dt / trajFinesse));
      for (let i = 0; i < trajectories.length; i++) {
        data.push({
          x: trajectories[i],
          y: tRefined,
          mode: 'lines',
          name: '',
          showlegend: false,
          line: { width: 0.5, color: '#333333' },
        });
      }
    }
    return data;
  }, [currentArr, trajectories, zArray, tArray, dt, trajFinesse, showTrajectories, detectorL]);

  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // ******* NEW: Arrival Time Distribution from Trajectories *******
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  const arrivalTimesData = React.useMemo(() => {
    if (!currentArr || currentArr.length === 0 || zArray.length === 0) return [];

    // Find the index closest to -L in zArray
    let detectorIndex = zArray.findIndex((z) => z >= -detectorL);
    if (detectorIndex === -1) detectorIndex = zArray.length - 1;

    // Get the probability current at the detector position
    const currentAtDetector = currentArr.map(row => row[detectorIndex]);

    return [
      {
        x: tArray,
        y: currentAtDetector,
        type: 'scatter',
        mode: 'lines',
        name: 'Simulated Arrival-Time Distribution (j at -L)',
        line: { color: 'blue' }
      }
    ];
  }, [currentArr, tArray, zArray, detectorL]);
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  /**
   * ----------------------------------
   * 8) Time Slider & Play/Pause Controls
   * ----------------------------------
   */
  const handleTimeSliderChange = (e) => {
    setTIndex(parseInt(e.target.value, 10));
  };

  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        setTIndex(prev => {
          if (!probArr || probArr.length === 0) return 0;
          return (prev + 1) % probArr.length;
        });
      };
      animationRef.current = setInterval(animate, 100);
      return () => clearInterval(animationRef.current);
    } else {
      clearInterval(animationRef.current);
    }
  }, [isPlaying, probArr]);

  /**
   * ----------------------------------
   * 9) Loading/Saving Simulations from IndexedDB
   * ----------------------------------
   */
  useEffect(() => {
    const loadSavedSims = async () => {
      try {
        const db = await initDB();
        const transaction = db.transaction(STORE_NAME, 'readonly');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.getAll();
        request.onsuccess = () => {
          setSavedSims(request.result);
        };
      } catch (error) {
        console.error('Error loading saved simulations:', error);
      }
    };
    loadSavedSims();
  }, []);

  const handleSaveSimulation = async () => {
    if (!probArr || probArr.length === 0) return;
    const simData = {
      name: simName,
      timestamp: Date.now(),
      barrierType,
      deltaAlpha,
      deltaZ0,
      gaussV0,
      gaussSigma,
      gaussZ0,
      gauss2V0,   // NEW: Save double Gaussian parameters
      gauss2Sigma,
      gauss2Z0,
      z0Packet,
      p0Packet,
      useSuperposition,    // NEW: Save superposition toggle
      z0Packet2,           // NEW: Save second wavefunction parameters
      p0Packet2,
      sigmaPacket2,
      xMin,
      xMax,
      Nx,
      nSteps,
      dt,
      zArray,
      tArray,
      probArr,
      currentArr,
    };
    try {
      const db = await initDB();
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      await store.add(simData);
      setSavedSims(prev => [...prev, simData]);
    } catch (error) {
      console.error('Error saving simulation:', error);
      alert('Failed to save simulation. Please try again.');
    }
  };

  const handleLoadSimulation = async (index) => {
    const rec = savedSims[index];
    setBarrierType(rec.barrierType);
    setDeltaAlpha(rec.deltaAlpha);
    setDeltaZ0(rec.deltaZ0);
    setGaussV0(rec.gaussV0);
    setGaussSigma(rec.gaussSigma);
    setGaussZ0(rec.gaussZ0);
    // NEW: Load double Gaussian parameters if available
    setGauss2V0(rec.gauss2V0 || 5.0);
    setGauss2Sigma(rec.gauss2Sigma || 0.5);
    setGauss2Z0(rec.gauss2Z0 || 5.0);
    setZ0Packet(rec.z0Packet);
    setP0Packet(rec.p0Packet);
    // NEW: Load superposition parameters
    setUseSuperposition(rec.useSuperposition || false);
    setZ0Packet2(rec.z0Packet2 || 5.0);
    setP0Packet2(rec.p0Packet2 || 0.0);
    setSigmaPacket2(rec.sigmaPacket2 || 1.0);
    setXMin(rec.xMin);
    setXMax(rec.xMax);
    setNx(rec.Nx);
    setNSteps(rec.nSteps);
    setDt(rec.dt);
    setZArray(rec.zArray);
    setTArray(rec.tArray);
    setProbArr(rec.probArr);
    setCurrentArr(rec.currentArr);
    setTIndex(0);
    // Optionally, re-compute trajectories if desired.
  };

  const handleDeleteSimulation = async (timestamp) => {
    try {
      const db = await initDB();
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      await store.delete(timestamp);
      setSavedSims(prev => prev.filter(sim => sim.timestamp !== timestamp));
    } catch (error) {
      console.error('Error deleting simulation:', error);
      alert('Failed to delete simulation. Please try again.');
    }
  };

  return (
    <div className="flex h-screen">
      {/* Static sidebars container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-1/5 min-w-[300px] border-r border-gray-300 p-4 overflow-y-auto">
          <div className="space-y-6">
            <div>
              <h1 classNamez="text-2xl font-bold mb-2">Quantum Simulator</h1>
              <p className="text-gray-600 text-sm mb-4">
                Simulate quantum wave packet dynamics, falling under gravity past a barrier.
              </p>
              <p className="text-gray-600 text-sm mb-4">
                Author: Adam 'Blvck' Blazejczak<br/>
                Promotor: Ward Wuyts<br/>
                Co-Promotor: Jef Hooyberghs<br/>
                Date: 2025-02-10
              </p>
              <button 
                onClick={() => setIsModalOpen(true)} 
                className="text-blue-500 hover:text-blue-600 text-sm"
              >
                Learn more
              </button>
            </div>
            {/* Potential Controls */}
            <div>
              <h2 className="text-xl font-bold mb-2">Potential</h2>
              <label className="block mb-2">
                <span className="font-semibold">Type:</span>
                <select 
                  value={barrierType} 
                  onChange={(e) => setBarrierType(e.target.value)}
                  className="ml-2 border p-1 rounded"
                >
                  <option value="delta">Delta Barrier</option>
                  <option value="gaussian">Gaussian Barrier</option>
                  <option value="doubleGaussian">Double Gaussian Barrier</option>
                </select>
              </label>
              {barrierType === 'delta' && (
                <div className="space-y-2">
                  <label className="block">
                    Alpha:
                    <input 
                      type="number" 
                      value={deltaAlpha} 
                      onChange={(e) => setDeltaAlpha(parseFloat(e.target.value))}
                      className="ml-2 border p-1 rounded w-20"
                    />
                  </label>
                  <label className="block">
                    z0:
                    <input 
                      type="number" 
                      value={deltaZ0} 
                      onChange={(e) => setDeltaZ0(parseFloat(e.target.value))}
                      className="ml-2 border p-1 rounded w-20"
                    />
                  </label>
                </div>
              )}
              {barrierType === 'gaussian' && (
                <div className="space-y-2">
                  <label className="block">
                    V0:
                    <input 
                      type="number" 
                      value={gaussV0} 
                      onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                      className="ml-2 border p-1 rounded w-20"
                    />
                  </label>
                  <label className="block">
                    sigma:
                    <input 
                      type="number" 
                      value={gaussSigma} 
                      onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                      className="ml-2 border p-1 rounded w-20"
                    />
                  </label>
                  <label className="block">
                    z0:
                    <input 
                      type="number" 
                      value={gaussZ0} 
                      onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                      className="ml-2 border p-1 rounded w-20"
                    />
                  </label>
                </div>
              )}
              {barrierType === 'doubleGaussian' && (
                <div className="space-y-4">
                  <div>
                    <h3 className="font-bold">Gaussian 1</h3>
                    <label className="block">
                      V0:
                      <input 
                        type="number" 
                        value={gaussV0} 
                        onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      sigma:
                      <input 
                        type="number" 
                        value={gaussSigma} 
                        onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      z0:
                      <input 
                        type="number" 
                        value={gaussZ0} 
                        onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                  </div>
                  <div>
                    <h3 className="font-bold">Gaussian 2</h3>
                    <label className="block">
                      V0:
                      <input 
                        type="number" 
                        value={gauss2V0} 
                        onChange={(e) => setGauss2V0(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      sigma:
                      <input 
                        type="number" 
                        value={gauss2Sigma} 
                        onChange={(e) => setGauss2Sigma(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      z0:
                      <input 
                        type="number" 
                        value={gauss2Z0} 
                        onChange={(e) => setGauss2Z0(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>
            {/* Wave Packet Controls */}
            <div>
              <h2 className="text-xl font-bold mb-2">Wave Packet Controls</h2>
              <div className="space-y-2">
                <label className="block">
                  First Wave Packet z0:
                  <input 
                    type="number"
                    value={z0Packet}
                    onChange={(e) => setZ0Packet(parseFloat(e.target.value))}
                    className="ml-2 border p-1 rounded w-20"
                  />
                </label>
                <label className="block">
                  First Wave Packet p0:
                  <input 
                    type="number"
                    value={p0Packet}
                    onChange={(e) => setP0Packet(parseFloat(e.target.value))}
                    className="ml-2 border p-1 rounded w-20"
                  />
                </label>
                <label className="block">
                  Use Superposition:
                  <input 
                    type="checkbox"
                    checked={useSuperposition}
                    onChange={(e) => setUseSuperposition(e.target.checked)}
                    className="ml-2"
                  />
                </label>
                {useSuperposition && (
                  <div className="ml-4 space-y-2">
                    <h3 className="font-bold">Second Wave Packet</h3>
                    <label className="block">
                      z0 (2nd):
                      <input 
                        type="number"
                        value={z0Packet2}
                        onChange={(e) => setZ0Packet2(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      p0 (2nd):
                      <input 
                        type="number"
                        value={p0Packet2}
                        onChange={(e) => setP0Packet2(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                    <label className="block">
                      sigma (2nd):
                      <input 
                        type="number"
                        value={sigmaPacket2}
                        onChange={(e) => setSigmaPacket2(parseFloat(e.target.value))}
                        className="ml-2 border p-1 rounded w-20"
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>
            {/* Save/Load Section */}
            <div className="border-t pt-4">
              <details className="cursor-pointer">
                <summary className="text-xl font-bold mb-2">Save & Load</summary>
                <div className="space-y-2 mt-2">
                  <input 
                    type="text"
                    value={simName}
                    onChange={(e) => setSimName(e.target.value)}
                    className="border p-1 rounded w-full"
                    placeholder="Simulation name"
                  />
                  <button 
                    onClick={handleSaveSimulation}
                    className="bg-blue-500 text-white px-3 py-1 rounded w-full"
                  >
                    Save Simulation
                  </button>
                  <h4 className="font-semibold mt-2">Saved Simulations:</h4>
                  <ul className="max-h-32 overflow-auto border p-2 rounded">
                    {savedSims.map((s, idx) => (
                      <li key={idx} className="flex justify-between items-center mb-1">
                        <div className="mr-2">{s.name}</div>
                        <div className="flex space-x-1">
                          <button 
                            onClick={() => handleLoadSimulation(idx)}
                            className="bg-green-400 text-white px-2 py-1 rounded"
                          >
                            Load
                          </button>
                          <button
                            onClick={() => handleDeleteSimulation(s.timestamp)}
                            className="bg-red-400 text-white px-2 py-1 rounded"
                          >
                            Del
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </details>
            </div>
          </div>
        </div>
        {/* Center Content */}
        <div className="flex-1 p-4 overflow-y-auto">
          <div className="space-y-4">
            <div className="flex-1">
              <div className="flex items-center mb-2">
                <h3 className="text-lg font-semibold">Probability Density Heatmap</h3>
                <InfoButton onClick={() => setActiveExplanation('probabilityDensity')} />
              </div>
              <Plot
                data={heatmapData}
                layout={{
                  width: undefined,
                  height: 400,
                  autosize: true,
                  margin: { t: 30, l: 50, r: 10, b: 40 },
                  xaxis: { title: 'z' },
                  yaxis: { title: 'time' },
                }}
                useResizeHandler
                style={{ width: '100%', height: '400px' }}
              />
            </div>
            <div className="flex-1 mt-4">
              <div className="flex items-center mb-2">
                <h3 className="text-lg font-semibold">Probability Density in 2D</h3>
                <InfoButton onClick={() => setActiveExplanation('probabilityCurrent')} />
              </div>
              <Plot
                data={lineData}
                layout={{
                  width: undefined,
                  height: 300,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { title: 'z', range: [xMin, xMax] },
                  yaxis: { title: '|ψ|²', side: 'left' },
                  yaxis2: { 
                    title: 'V(z)', 
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false,
                  },
                  showlegend: true,
                  legend: { x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' },
                }}
                useResizeHandler
                style={{ width: '100%', height: '300px' }}
              />
            </div>
            <div className="flex-1 mt-4">
              <div className="flex items-center mb-2">
                <h3 className="text-lg font-semibold">Probability Current & Bohmian Trajectories</h3>
                <InfoButton onClick={() => setActiveExplanation('bohmianTrajectories')} />
              </div>
              <Plot
                data={trajectoryData}
                layout={{
                  width: undefined,
                  height: 400,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { title: 'z' },
                  yaxis: { title: 'time' },
                  showlegend: true,
                }}
                useResizeHandler
                style={{ width: '100%', height: '400px' }}
              />
            </div>
            <div className="flex-1 mt-4">
              <div className="flex items-center mb-2">
                <h3 className="text-lg font-semibold">Time of Arrival From Paper</h3>
                <InfoButton onClick={() => setActiveExplanation('potentialEnergy')} />
              </div>
              <Plot
                data={arrivalTimeData}
                layout={{
                  width: undefined,
                  height: 300,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { title: 'Time τ' },
                  yaxis: { title: 'Density / Current' },
                  showlegend: true,
                  legend: { x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' },
                }}
                useResizeHandler
                style={{ width: '100%', height: '300px' }}
              />
            </div>
            <div className="flex-1 mt-4">

              <div className="flex items-center mb-2">
                <h3 className="text-lg font-semibold">Arrival Time Distribution from Probability Current</h3>
                <InfoButton onClick={() => setActiveExplanation('arrivalTimeDistribution')} />
              </div>
              
              <Plot
                data={arrivalTimesData}
                layout={{
                  width: undefined,
                  height: 300,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { title: 'Arrival Time (τ)' },
                  yaxis: { title: 'Counts / Probability' },
                  showlegend: true,
                  legend: { x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' },
                }}
                useResizeHandler
                style={{ width: '100%', height: '300px' }}
              />
            </div>
          </div>
        </div>
        {/* Right Sidebar */}
        <div className="w-1/5 min-w-[250px] border-l border-gray-300 p-4 overflow-y-auto">
          <div className="space-y-4">
            <h2 className="text-xl font-bold">Simulation Controls</h2>
            <div className="space-y-2">
              <label className="block">
                xMin:
                <input 
                  type="number"
                  value={xMin}
                  onChange={(e) => setXMin(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-16"
                />
              </label>
              <label className="block">
                xMax:
                <input 
                  type="number"
                  value={xMax}
                  onChange={(e) => setXMax(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-16"
                />
              </label>
              <label className="block">
                Nx:
                <input 
                  type="number"
                  value={Nx}
                  onChange={(e) => setNx(parseInt(e.target.value))}
                  className="ml-2 border p-1 rounded w-16"
                />
              </label>
              <label className="block">
                nSteps:
                <input 
                  type="number"
                  value={nSteps}
                  onChange={(e) => setNSteps(parseInt(e.target.value))}
                  className="ml-2 border p-1 rounded w-16"
                />
              </label>
              <label className="block">
                dt:
                <input 
                  type="number"
                  value={dt}
                  onChange={(e) => setDt(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-16"
                  step="0.001"
                />
              </label>
              <label className="block">
                Detector at -L:
                <input 
                  type="number"
                  value={detectorL}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) setDetectorL(val);
                  }}
                  className="ml-2 border p-1 rounded w-20"
                  step="any"
                />
              </label>
              {/* Particle Trajectory Parameters */}
              <label className="block">
                Bohmian Trajectory Spawn Center:
                <input 
                  type="number"
                  value={particleSpawnCenter}
                  onChange={(e) => setParticleSpawnCenter(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-20"
                />
              </label>
              <label className="block">
                Bohmian Trajectory Spawn Width:
                <input 
                  type="number"
                  value={particleSpawnWidth}
                  onChange={(e) => setParticleSpawnWidth(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-20"
                />
              </label>
              <label className="block">
                Num Particles:
                <input 
                  type="number"
                  value={numParticles}
                  onChange={(e) => setNumParticles(parseInt(e.target.value))}
                  className="ml-2 border p-1 rounded w-20"
                />
              </label>
              {/* NEW: Trajectory Integration Factor Control */}
              {/* <label className="block">
                Trajectory Integration Factor:
                <input 
                  type="number"
                  value={trajIntegrationFactor}
                  onChange={(e) => setTrajIntegrationFactor(parseFloat(e.target.value))}
                  className="ml-2 border p-1 rounded w-20"
                  step="0.1"
                />
                <small className="block text-xs text-gray-600">
                  Lower this value for finer integration.
                </small>
              </label> */}
              <label className="block">
                Show Trajectories:
                <input 
                  type="checkbox"
                  checked={showTrajectories}
                  onChange={(e) => setShowTrajectories(e.target.checked)}
                  className="ml-2"
                />
              </label>
            </div>
            <button 
              onClick={handleRunSimulation}
              className="bg-green-500 text-white px-3 py-1 rounded"
            >
              Run Simulation
            </button>
            <div className="mt-4">
              <h3 className="font-semibold">Time Index:</h3>
              <input 
                type="range"
                min="0"
                max={(probArr.length - 1) || 0}
                value={tIndex}
                onChange={handleTimeSliderChange}
                disabled={!probArr || probArr.length === 0}
              />
              <div className="flex space-x-2 mt-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="bg-blue-500 text-white px-3 py-1 rounded"
                  disabled={!probArr || probArr.length === 0}
                >
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <span>T = {tArray[tIndex] ? tArray[tIndex].toFixed(3) : 0}</span>
              </div>
            </div>
            <div className="mt-4">
              <h3 className="font-semibold">Wavefunction Distribution:</h3>
              <p>x &lt; 0: {(() => {
                if (!probArr || zArray.length < 2) return 0;
                const slice = probArr[tIndex] || [];
                const dz = zArray[1] - zArray[0];
                let leftSum = 0, rightSum = 0;
                for (let i = 0; i < zArray.length; i++) {
                  if (zArray[i] < 0) leftSum += slice[i];
                  else rightSum += slice[i];
                }
                const total = leftSum + rightSum;
                return total ? (leftSum / total * 100).toFixed(2) : 0;
              })()}%
              </p>
              <p>x ≥ 0: {(() => {
                if (!probArr || zArray.length < 2) return 0;
                const slice = probArr[tIndex] || [];
                const dz = zArray[1] - zArray[0];
                let leftSum = 0, rightSum = 0;
                for (let i = 0; i < zArray.length; i++) {
                  if (zArray[i] < 0) leftSum += slice[i];
                  else rightSum += slice[i];
                }
                const total = leftSum + rightSum;
                return total ? (rightSum / total * 100).toFixed(2) : 0;
              })()}%
              </p>
            </div>
          </div>
        </div>
      </div>
      <InfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      <ExplanationModal
        isOpen={!!activeExplanation}
        onClose={() => setActiveExplanation(null)}
        content={activeExplanation ? explanations[activeExplanation] : ''}
      />
    </div>
  );
};

export default Dashboard;
