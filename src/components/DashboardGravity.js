import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import { fft, ifft } from 'fft-js';
import { debounce } from 'lodash';

/**
 * Example InfoModal component, if you want to keep something similar:
 */
const InfoModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-4 rounded shadow-md max-w-lg w-full">
        <h2 className="text-xl font-bold mb-2">Simulation Info</h2>
        <p className="mb-4">
          This dashboard simulates the time evolution of a wavefunction under
          a given 1D potential using the split-operator method.
        </p>
        <button 
          onClick={onClose} 
          className="bg-blue-500 text-white px-4 py-2 rounded"
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
  const [barrierType, setBarrierType] = useState('delta'); // 'delta' | 'gaussian'
  
  // Delta-barrier parameters
  const [deltaAlpha, setDeltaAlpha] = useState(15.0);
  const [deltaZ0, setDeltaZ0] = useState(0.0);

  // Gaussian-barrier parameters
  const [gaussV0, setGaussV0] = useState(5.0);
  const [gaussSigma, setGaussSigma] = useState(0.5);
  const [gaussZ0, setGaussZ0] = useState(0.0);

  // Wave packet initial conditions
  const [z0Packet, setZ0Packet] = useState(10.0);
  const [p0Packet, setP0Packet] = useState(0.0);

  // PDE solver parameters (Note: xMin/xMax here are the plotting limits)
  const [xMin, setXMin] = useState(-20);
  const [xMax, setXMax] = useState(20);
  const [Nx, setNx] = useState(256);
  const [nSteps, setNSteps] = useState(200);
  const [dt, setDt] = useState(0.01);

  // Gravity constants (hard-coded for now)
  const hbar = 1.0;
  const m = 1.0;
  const g = 1.0;

  // Storage for simulation results
  const [zArray, setZArray] = useState([]); // now simulation grid (extended)
  const [tArray, setTArray] = useState([]);
  const [probArr, setProbArr] = useState([]); // shape: [nSteps, Nx]

  // The currently selected time index (for the single-slice line plot):
  const [tIndex, setTIndex] = useState(0);

  // For playing/pausing time slider:
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef(null);

  // For localStorage or naming the simulation
  const [simName, setSimName] = useState('My Simulation');
  const [savedSims, setSavedSims] = useState([]);

  // Info Modal
  const [isModalOpen, setIsModalOpen] = useState(false);

  /**
   * ---------------------------------------
   * 2) Potential Helper Functions (in JS)
   * ---------------------------------------
   */
  // Gravity potential V_grav(z) = m*g*z
  const gravityPotential = (z) => m * g * z;

  // For "delta" barrier, we approximate with a narrow Gaussian
  const deltaBarrier = (z, z0, alpha, dz) => {
    // Use sigma ~ dz/2 for the Gaussian representation
    const sigma = dz / 2;
    return alpha * Math.exp( -((z - z0)**2)/(2*sigma**2) ) / (Math.sqrt(2*Math.PI)*sigma);
  };

  // True Gaussian barrier
  const gaussianBarrier = (z, z0, V0, sigma) => {
    return V0 * Math.exp( -((z - z0)**2)/(2*sigma**2) );
  };

  // Initial wavefunction: Gaussian wave packet
  const initialWavefunction = (z, z0, p0, sigma0=1.0) => {
    // 1D normalized Gaussian wave packet
    const normFactor = 1.0 / Math.pow(2.0 * Math.PI * sigma0**2, 0.25);
    const phase = (p0 * (z - z0)) / hbar; 
    const realPart = normFactor * Math.exp( -((z - z0)**2) / (4.0 * sigma0**2) ) * Math.cos(phase);
    const imagPart = normFactor * Math.exp( -((z - z0)**2) / (4.0 * sigma0**2) ) * Math.sin(phase);
    return { re: realPart, im: imagPart };
  };

  /**
   * --------------------------------------
   * 3) Split-Operator Step in JavaScript
   * --------------------------------------
   */
  // Multiply wavefunction by exp(-i * V * dt / (2*hbar))
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

  // Multiply in momentum space by exp(-i * (hbar*k^2/(2m)) * dt / hbar)
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

  // Optionally, you can add an absorbing mask (if needed) here.

  /**
   * ------------------------------------------
   * 4) The "simulateN" function in JavaScript
   * ------------------------------------------
   */
  const simulateN = () => {
    // Extend the simulation domain.
    // The plotting range is [xMin, xMax], but we simulate on a range 1.5 times wider.
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

    // 4.2) Build the potential array
    let barrierArr = new Array(Nx).fill(0);
    if (barrierType === 'delta') {
      for (let i = 0; i < Nx; i++) {
        barrierArr[i] = deltaBarrier(zArr[i], deltaZ0, deltaAlpha, dz);
      }
    } else {
      for (let i = 0; i < Nx; i++) {
        barrierArr[i] = gaussianBarrier(zArr[i], gaussZ0, gaussV0, gaussSigma);
      }
    }
    // Add gravity
    for (let i = 0; i < Nx; i++) {
      barrierArr[i] += gravityPotential(zArr[i]);
    }

    // 4.3) Build the k-array for momentum space
    const kArr = new Array(Nx);
    for (let i = 0; i < Nx; i++) {
      const kIndex = (i < Nx/2) ? i : i - Nx;
      kArr[i] = (2.0 * Math.PI * kIndex) / (Nx * dz);
    }

    // 4.4) Build the kinetic factor for a full step in momentum space
    const kineticRe = new Array(Nx);
    const kineticIm = new Array(Nx);
    for (let i = 0; i < Nx; i++) {
      const phase = -(kArr[i]*kArr[i] * dt)/(2*m);
      kineticRe[i] = Math.cos(phase);
      kineticIm[i] = Math.sin(phase);
    }

    // 4.5) Initialize the wavefunction
    const psiRe = new Array(Nx);
    const psiIm = new Array(Nx);
    for (let i = 0; i < Nx; i++) {
      const { re, im } = initialWavefunction(zArr[i], z0Packet, p0Packet, 1.0);
      psiRe[i] = re;
      psiIm[i] = im;
    }

    // 4.6) Setup array to store probability density at each step
    const probStorage = [];
    probStorage.length = nSteps;

    // 4.7) Setup temporary FFT arrays (not strictly necessary if reusing arrays)
    const inputRe = new Float64Array(Nx);
    const inputIm = new Float64Array(Nx);
    const outputRe = new Float64Array(Nx);
    const outputIm = new Float64Array(Nx);

    const dtOver2hbar = dt / (2*hbar);

    // 4.8) Main time evolution loop
    for (let step = 0; step < nSteps; step++) {
      // Store current probability distribution
      const probRow = new Array(Nx);
      for (let i = 0; i < Nx; i++) {
        probRow[i] = psiRe[i]*psiRe[i] + psiIm[i]*psiIm[i];
      }
      probStorage[step] = probRow;

      // First half-step in potential
      applyPotentialHalfStep(psiRe, psiIm, barrierArr, dtOver2hbar);

      // Full kinetic step:
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

      // (Optionally, you could apply an absorbing mask here.)
      applyAbsorbingMask(psiRe, psiIm, zArr);
    }

    // 4.9) Build time array
    const tArr = new Array(nSteps);
    for (let i = 0; i < nSteps; i++) {
      tArr[i] = i * dt;
    }

    return { zArr, tArr, probArr: probStorage };
  };

  /**
   * ------------------------------------------------
   * 5) Handling the "Run Simulation" button
   * ------------------------------------------------
   */
  const handleRunSimulation = () => {
    const { zArr, tArr, probArr } = simulateN();
    setZArray(zArr);
    setTArray(tArr);
    setProbArr(probArr);
    setTIndex(0);
  };

  /**
   * -----------------------------------------------------------
   * 6) Plotly Data for the Heatmap and Single-Slice Line
   * -----------------------------------------------------------
   */
  // Heatmap uses the full simulation domain
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
    }];
  }, [probArr, zArray, tArray]);

  // Line plot: Filter to only show points in the plotting range [xMin, xMax]
  const lineData = React.useMemo(() => {
    if (!probArr || probArr.length === 0) return [];

    const slice = probArr[tIndex] || [];
    // Compute potential values on the full simulation domain.
    const potentialValues = zArray.map(z => {
      let barrier = 0;
      if (barrierType === 'delta') {
        barrier = deltaBarrier(z, deltaZ0, deltaAlpha, (xMax - xMin) / Nx);
      } else {
        barrier = gaussianBarrier(z, gaussZ0, gaussV0, gaussSigma);
      }
      return barrier + gravityPotential(z);
    });

    // Only include points whose z value is between xMin and xMax.
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
  }, [probArr, tIndex, zArray, barrierType, deltaZ0, deltaAlpha, gaussZ0, gaussV0, gaussSigma, xMin, xMax, Nx]);

  /**
   * --------------------------------------------------
   * 7) Indicators: Percentage of Wavefunction in x<0 and x>=0
   * --------------------------------------------------
   */
  const indicatorPercentages = React.useMemo(() => {
    if (!probArr || probArr.length === 0 || zArray.length < 2) return { left: 0, right: 0 };
    const slice = probArr[tIndex] || [];
    const dz = zArray[1] - zArray[0]; // assume uniform spacing
    let leftSum = 0, rightSum = 0;
    for (let i = 0; i < zArray.length; i++) {
      if (zArray[i] < 0) {
        leftSum += slice[i];
      } else {
        rightSum += slice[i];
      }
    }
    const total = leftSum + rightSum;
    return {
      left: total > 0 ? (leftSum / total) * 100 : 0,
      right: total > 0 ? (rightSum / total) * 100 : 0,
    };
  }, [probArr, zArray, tIndex]);

  /**
   * --------------------------------------------------
   * 8) Time Slider & Play/Pause Controls
   * --------------------------------------------------
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
   * --------------------------------------------------
   * 9) Loading/Saving Simulations from IndexedDB
   * --------------------------------------------------
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

 // Absorbing mask to damp the wavefunction near the boundaries
const applyAbsorbingMask = (psiRe, psiIm, zArr) => {
  // Define a margin (e.g. 10% of the total domain on each side)
  const margin = 0.1 * (xMax - xMin);
  for (let i = 0; i < psiRe.length; i++) {
    let factor = 1;
    if (zArr[i] < xMin + margin) {
      // Dampen smoothly at the left edge
      factor = Math.exp(- Math.pow((xMin + margin - zArr[i]) / margin, 2));
    } else if (zArr[i] > xMax - margin) {
      // Dampen smoothly at the right edge
      factor = Math.exp(- Math.pow((zArr[i] - (xMax - margin)) / margin, 2));
    }
    psiRe[i] *= factor;
    psiIm[i] *= factor;
  }
};

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
      z0Packet,
      p0Packet,
      xMin,
      xMax,
      Nx,
      nSteps,
      dt,
      zArray,
      tArray,
      probArr,
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
    setZ0Packet(rec.z0Packet);
    setP0Packet(rec.p0Packet);
    setXMin(rec.xMin);
    setXMax(rec.xMax);
    setNx(rec.Nx);
    setNSteps(rec.nSteps);
    setDt(rec.dt);
    setZArray(rec.zArray);
    setTArray(rec.tArray);
    setProbArr(rec.probArr);
    setTIndex(0);
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
      {/* Info Button */}
      <button 
        onClick={() => setIsModalOpen(true)} 
        className="absolute top-4 left-4 bg-blue-500 text-white p-2 rounded z-10"
      >
        Info
      </button>
      <InfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />

      {/* -------------- Left Pane -------------- */}
      <div className="w-1/5 border-r border-gray-300 p-4 flex flex-col space-y-4">
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

        <h2 className="text-xl font-bold mt-4 mb-2">Wave Packet</h2>
        <div className="space-y-2">
          <label className="block">
            z0:
            <input 
              type="number"
              value={z0Packet}
              onChange={(e) => setZ0Packet(parseFloat(e.target.value))}
              className="ml-2 border p-1 rounded w-20"
            />
          </label>
          <label className="block">
            p0:
            <input 
              type="number"
              value={p0Packet}
              onChange={(e) => setP0Packet(parseFloat(e.target.value))}
              className="ml-2 border p-1 rounded w-20"
            />
          </label>
        </div>
      </div>

      {/* -------------- Center Pane -------------- */}
      <div className="w-3/5 border-r border-gray-300 p-4 flex flex-col">
        <div className="flex-1">
          <h2 className="text-xl font-bold mb-2">Heatmap of |ψ(z,t)|²</h2>
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
          <h2 className="text-xl font-bold mb-2">
            Slice at tIndex = {tIndex} (t = {tArray[tIndex] ? tArray[tIndex].toFixed(3) : 0})
          </h2>
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
              legend: { 
                x: 0.5,
                y: 1.15,
                xanchor: 'center',
                orientation: 'h',
              },
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
          />
        </div>
      </div>

      {/* -------------- Right Pane -------------- */}
      <div className="w-1/5 p-4 space-y-4 flex flex-col">
        <h2 className="text-xl font-bold">Simulation</h2>
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
        </div>
        
        <button 
          onClick={handleRunSimulation}
          className="bg-green-500 text-white px-3 py-1 rounded"
        >
          Run Simulation
        </button>

        {/* Time Slider + Play/Pause */}
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

        {/* New Indicators */}
        <div className="mt-4">
          <h3 className="font-semibold">Wavefunction Distribution:</h3>
          <p>x &lt; 0: {indicatorPercentages.left.toFixed(2)}%</p>
          <p>x ≥ 0: {indicatorPercentages.right.toFixed(2)}%</p>
        </div>

        {/* Save / Load Simulations */}
        <div className="mt-4 space-y-2">
          <h3 className="font-semibold">Save / Load</h3>
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
            Save to IndexedDB
          </button>

          <h4 className="font-semibold mt-2">Saved Sims:</h4>
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
      </div>
    </div>
  );
};

export default Dashboard;
