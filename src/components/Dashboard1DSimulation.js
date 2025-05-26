import React, { useState, useEffect, useRef, createContext, useContext } from 'react';
import Plot from 'react-plotly.js';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { explanations, InfoButton } from './explanations';

import { hbar, m, g } from '../physics/constants';

import {
  simulateN
} from '../physics/quantumSplitOperator1D';

// Add import for plotting functions
import { 
  generateHeatmapData,
  generateLineData,
  generateArrivalTimeData,
  generateTrajectoryData,
  generateArrivalTimesData,
  interpolateVelocity
} from '../physics/plotting';

// Add new imports for icons (you'll need to install @heroicons/react)
import { Bars3Icon, XMarkIcon, SunIcon, MoonIcon } from '@heroicons/react/24/outline';

// Create Theme Context
const ThemeContext = createContext({
  isDark: false,
  toggleTheme: () => {},
});

// Theme Provider Component
const ThemeProvider = ({ children }) => {
  const [isDark, setIsDark] = useState(() => {
    // Check local storage or system preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) return savedTheme === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    // Update document class and local storage when theme changes
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

/**
 * Example InfoModal component
 */
const InfoModal = ({ isOpen, onClose, isDark }) => {
  if (!isOpen) return null;

  const markdown = `# Quantum Simulation Info

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

This simulation uses the [Split-Operator Method](https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html) to evolve the wavefunction in time. $|\\psi|^2$ is plotted as a heatmap and in 1D for visual inspection.
Since the Split-Operator Method assumes periodic boundary conditions, we apply exponential decay at about 10\% of the boundaries, to avoid unwanted reflections.

## Running The Simulation

Choose a potential parameter, initial conditions, and simulation parameters. After this click on the green "Run Simulation" button.
Simulations can be stored (locally in browser, including simulation data) in the left sidebar.

For more information about each graph, click on the info button next to each graph.
  `;

  return (
    <div className={`fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50 ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <div className={`bg-white p-6 rounded shadow-md max-w-2xl w-full overflow-y-auto max-h-[90vh] ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            className={`space-y-4 text-${isDark ? 'gray-300' : 'gray-700'}`}
            components={{
              h1: ({node, ...props}) => <h1 className="text-3xl font-bold my-4" {...props} />,
              h2: ({node, ...props}) => <h2 className="text-2xl font-bold my-3" {...props} />,
              h3: ({node, ...props}) => <h3 className="text-xl font-bold my-2" {...props} />,
              h4: ({node, ...props}) => <h4 className="text-lg font-bold my-2" {...props} />,
              p: ({node, ...props}) => <p className="my-2" {...props} />,
              a: ({node, ...props}) => <a className="text-blue-500 hover:text-blue-600" {...props} />
            }}
          >
            {markdown}
          </ReactMarkdown>
        </div>
        <button 
          onClick={onClose} 
          className={`mt-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors ${isDark ? 'bg-gray-700' : 'bg-gray-800'}`}
        >
          Close
        </button>
      </div>
    </div>
  );
};

const ExplanationModal = ({ isOpen, onClose, content, isDark }) => {
  if (!isOpen) return null;

  return (
    <div className={`fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50 ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <div className={`bg-white p-6 rounded shadow-md max-w-2xl w-full overflow-y-auto max-h-[90vh] ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            className={`space-y-4 text-${isDark ? 'gray-300' : 'gray-700'}`}
            components={{
              h1: ({node, ...props}) => <h1 className="text-3xl font-bold my-4" {...props} />,
              h2: ({node, ...props}) => <h2 className="text-2xl font-bold my-3" {...props} />,
              h3: ({node, ...props}) => <h3 className="text-xl font-bold my-2" {...props} />,
              h4: ({node, ...props}) => <h4 className="text-lg font-bold my-2" {...props} />,
              p: ({node, ...props}) => <p className="my-2" {...props} />,
              a: ({node, ...props}) => <a className="text-blue-500 hover:text-blue-600" {...props} />
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
        <button 
          onClick={onClose} 
          className={`mt-6 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors ${isDark ? 'bg-gray-700' : 'bg-gray-800'}`}
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
  const { isDark, toggleTheme } = useContext(ThemeContext);

  /**
   * -------------------------------
   * 1) State for PDE / Potential
   * -------------------------------
   */
  // Available potential types: 'delta', 'gaussian', 'doubleGaussian', 'squareWell'
  const [barrierType, setBarrierType] = useState('gaussian'); 
  
  // Delta-barrier parameters
  const [deltaAlpha, setDeltaAlpha] = useState(8.0); // strength of the barrier - gamme in Siddhant Das Paper
  const [deltaZ0, setDeltaZ0] = useState(0.0);

  // Gaussian-barrier parameters (for single Gaussian)
  const [gaussV0, setGaussV0] = useState(10.0);
  const [gaussSigma, setGaussSigma] = useState(1);
  const [gaussZ0, setGaussZ0] = useState(0.0);

  // NEW: Parameters for the second Gaussian (for doubleGaussian potential)
  const [gauss2V0, setGauss2V0] = useState(5.0);
  const [gauss2Sigma, setGauss2Sigma] = useState(0.5);
  const [gauss2Z0, setGauss2Z0] = useState(5.0);

  // Wave packet initial conditions for the first wavefunction
  const [z0Packet, setZ0Packet] = useState(10.0);
  const [p0Packet, setP0Packet] = useState(0.0);
  const [sigmaPacket, setSigmaPacket] = useState(1.0);

  // NEW: Additional state for a second wavefunction (for superposition)
  const [useSuperposition, setUseSuperposition] = useState(false);
  const [z0Packet2, setZ0Packet2] = useState(5.0);
  const [p0Packet2, setP0Packet2] = useState(0.0);
  const [sigmaPacket2, setSigmaPacket2] = useState(1.0);

  // PDE solver parameters (xMin/xMax are the plotting limits)
  const [xMin, setXMin] = useState(-10);
  const [xMax, setXMax] = useState(20);
  const [Nx, setNx] = useState(512);
  const [nSteps, setNSteps] = useState(5000);
  const [dt, setDt] = useState(0.01);

  // Detector parameter L (the detector is at z = –L, L > 0)
  const [detectorL, setDetectorL] = useState(5.0);

  // New state for storing simulation results:
  const [zArray, setZArray] = useState([]);       // simulation grid
  const [tArray, setTArray] = useState([]);         // simulation time array
  const [probArr, setProbArr] = useState([]);       // probability density [nSteps x Nx]
  const [currentArr, setCurrentArr] = useState([]); // probability current [nSteps x Nx]
  const [cumulativeLossArrLeft, setCumulativeLossArrLeft] = useState([]); // cumulative probability loss left [nSteps]
  const [cumulativeLossArrRight, setCumulativeLossArrRight] = useState([]); // cumulative probability loss right [nSteps]

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

  // Add these new state variables near the top with other states
  const [isSimulating, setIsSimulating] = useState(false);
  const [progress, setProgress] = useState(0);

  // Add state for mobile menu
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  /**
   * ---------------------------------------
   * 5) Helper Functions for Particle Trajectories
   * ---------------------------------------
   *
   * We use a simple Euler integration for trajectories using v(z,t) = j(z,t)/|ψ(z,t)|².
   * The parameter subSteps (trajectory finesse) subdivides each dt.
   * 
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
  const handleRunSimulation = async () => {
    setIsSimulating(true);
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      const { zArr, tArr, probArr, currentArr, cumulativeLossArrLeft, cumulativeLossArrRight } = simulateN({
        xMin, xMax, Nx, nSteps, dt, hbar, m, barrierType, deltaZ0, deltaAlpha,
        gaussZ0, gaussV0, gaussSigma, gauss2Z0, gauss2V0, gauss2Sigma, useSuperposition,
        z0Packet, p0Packet, z0Packet2, p0Packet2, sigmaPacket2
      });
      
      setZArray(zArr);
      setTArray(tArr);
      setProbArr(probArr);
      setCurrentArr(currentArr);
      setCumulativeLossArrLeft(cumulativeLossArrLeft);
      setCumulativeLossArrRight(cumulativeLossArrRight);
      setTIndex(0);
      
      const trajs = computeTrajectories(probArr, currentArr, zArr, tArr, particleSpawnCenter, particleSpawnWidth, numParticles, trajFinesse, trajIntegrationFactor);
      setTrajectories(trajs);

      // Small delay before hiding the loading state
      setTimeout(() => {
        setIsSimulating(false);
      }, 500);

    } catch (error) {
      console.error('Simulation failed:', error);
      setIsSimulating(false);
    }
  };

  /**
   * -------------------------------------------
   * 7) Plotly Data for the Different Plots
   * -------------------------------------------
   */
  // Heatmap: |ψ(z,t)|²
  const heatmapData = React.useMemo(() => 
    generateHeatmapData(probArr, zArray, tArray),
    [probArr, zArray, tArray]
  );

  // Slice plot (at the selected tIndex)
  const lineData = React.useMemo(() => 
    generateLineData(probArr, tIndex, zArray, barrierType, {
      deltaZ0, deltaAlpha, gaussZ0, gaussV0, gaussSigma,
      gauss2Z0, gauss2V0, gauss2Sigma, xMin, xMax, Nx
    }),
    [probArr, tIndex, zArray, barrierType, deltaZ0, deltaAlpha,
     gaussZ0, gaussV0, gaussSigma, gauss2Z0, gauss2V0, gauss2Sigma,
     xMin, xMax, Nx]
  );

  // Arrival Time Density & Probability Current (analytic) plot
  const arrivalTimeData = React.useMemo(() => 
    generateArrivalTimeData({
      deltaAlpha, detectorL, z0Packet, p0Packet,
      sigmaPacket, xMin, xMax, Nx
    }),
    [deltaAlpha, detectorL, z0Packet, p0Packet,
     sigmaPacket, xMin, xMax, Nx]
  );

  // Particle Trajectories over the Probability Current heatmap
  const trajectoryData = React.useMemo(() => 
    generateTrajectoryData({
      currentArr, trajectories, zArray, tArray,
      dt, trajFinesse, showTrajectories, detectorL
    }),
    [currentArr, trajectories, zArray, tArray,
     dt, trajFinesse, showTrajectories, detectorL]
  );

  // ******* Arrival Time Distribution from probability current *******
  const arrivalTimesData = React.useMemo(() => 
    generateArrivalTimesData(currentArr, tArray, zArray, detectorL),
    [currentArr, tArray, zArray, detectorL]
  );

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
      // Convert dt to milliseconds for the interval
      // dt is in seconds, so multiply by 1000 to get milliseconds
      const intervalTime = dt * 1000;
      animationRef.current = setInterval(animate, intervalTime);
      return () => clearInterval(animationRef.current);
    } else {
      clearInterval(animationRef.current);
    }
  }, [isPlaying, probArr, dt]); // Add dt to dependencies

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

  // Update plot themes based on dark mode
  const getPlotTheme = () => ({
    plot_bgcolor: isDark ? '#1f2937' : '#ffffff',
    paper_bgcolor: isDark ? '#1f2937' : '#ffffff',
    font: {
      color: isDark ? '#ffffff' : '#000000',
    },
    xaxis: {
      gridcolor: isDark ? '#374151' : '#e5e7eb',
      zerolinecolor: isDark ? '#374151' : '#e5e7eb',
    },
    yaxis: {
      gridcolor: isDark ? '#374151' : '#e5e7eb',
      zerolinecolor: isDark ? '#374151' : '#e5e7eb',
    },
  });

  return (
    <div className={`flex flex-col min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Mobile Header */}
      <div className={`lg:hidden flex items-center justify-between p-4 ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b`}>
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className={`p-2 rounded-md ${isDark ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          {isMobileMenuOpen ? <XMarkIcon className="h-6 w-6" /> : <Bars3Icon className="h-6 w-6" />}
        </button>
        <h1 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Quantum Arrival Time
        </h1>
        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          className={`p-2 rounded-md ${isDark ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          {isDark ? <SunIcon className="h-6 w-6" /> : <MoonIcon className="h-6 w-6" />}
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar - Collapsible on mobile */}
        <div className={`
          fixed lg:relative lg:flex
          ${isMobileMenuOpen ? 'flex' : 'hidden'}
          z-40 ${isDark ? 'bg-gray-800' : 'bg-white'}
          w-full lg:w-1/5 lg:min-w-[300px]
          h-full overflow-y-auto
          border-r ${isDark ? 'border-gray-700' : 'border-gray-300'}
        `}>
          <div className="flex flex-col flex-1 p-4 space-y-6">
            {/* Add theme toggle at the top of sidebar */}
            <div className="flex justify-between items-center">
              <h1 className={`hidden lg:visible text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Quantum Arrival Time
              </h1>
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-md ${
                  isDark 
                    ? 'text-gray-300 hover:bg-gray-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
                title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              >
                {isDark ? (
                  <SunIcon className="h-6 w-6" />
                ) : (
                  <MoonIcon className="h-6 w-6" />
                )}
              </button>
            </div>
            
            {/* Main content wrapper with bottom padding */}
            <div className="pb-48">
              <div className="space-y-6">
                <div>
                  <h1 className={`lg:hidden text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    Quantum Arrival Time
                  </h1>
                  <p className={`text-${isDark ? 'gray-300' : 'gray-600'} text-sm mb-2`}>
                    Simulate quantum wave packet dynamics, falling under gravity past a barrier.
                  </p>
                  <button 
                    onClick={() => setIsModalOpen(true)} 
                    className={`text-${isDark ? 'gray-300' : 'blue-500'} hover:text-${isDark ? 'gray-400' : 'blue-600'} text-sm pb-2 mb-2`}
                  >
                    Learn more
                  </button>
                  <p className={`border-t pt-2 ${isDark ? 'text-gray-300' : 'text-gray-600'} text-sm mb-4`}>
                    Author: Adam 'Blvck' Blazejczak<br/>
                    Promotor: Ward Struyve<br/>
                    Co-Promotor: Jef Hooyberghs<br/>
                    <br/>
                    Bachelor Thesis Extra Credit Project<br/>
                    University of Hasselt & KU Leuven<br/>
                    Date: 2025-05-27
                  </p>
                </div>
                {/* Potential Controls */}
                <div className={`visible lg:hidden p-4 border-b ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                  <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Simulation Controls</h2>
                  <div className="space-y-2">
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      xMin:
                      <input 
                        type="number"
                        value={xMin}
                        onChange={(e) => setXMin(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-16 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        step="any"
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      xMax:
                      <input 
                        type="number"
                        value={xMax}
                        onChange={(e) => setXMax(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-16 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Nx:
                      <div className="flex items-center gap-2">
                        <input 
                          type="number"
                          value={Nx}
                          onChange={(e) => setNx(parseInt(e.target.value))}
                          className={`ml-2 border p-1 rounded w-16 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        />
                        <button
                          onClick={() => setNx(prev => Math.max(2, prev / 2))}
                          className={`px-2 py-1 text-sm rounded ${isDark ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-900'}`}
                          title="Halve Nx"
                        >
                          /2
                        </button>
                        <button
                          onClick={() => setNx(prev => prev * 2)}
                          className={`px-2 py-1 text-sm rounded ${isDark ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-200 hover:bg-gray-300 text-gray-900'}`}
                          title="Double Nx"
                        >
                          *2
                        </button>
                      </div>
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      nSteps:
                      <input 
                        type="number"
                        value={nSteps}
                        onChange={(e) => setNSteps(parseInt(e.target.value))}
                        className={`ml-2 border p-1 rounded w-16 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      dt:
                      <input 
                        type="number"
                        value={dt}
                        onChange={(e) => setDt(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-16 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        step="0.001"
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Detector at -L:
                      <input 
                        type="number"
                        value={detectorL}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) setDetectorL(val);
                        }}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        step="any"
                      />
                    </label>
                    {/* Particle Trajectory Parameters */}
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Bohmian Trajectory Spawn Center:
                      <input 
                        type="number"
                        value={particleSpawnCenter}
                        onChange={(e) => setParticleSpawnCenter(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Bohmian Trajectory Spawn Width:
                      <input 
                        type="number"
                        value={particleSpawnWidth}
                        onChange={(e) => setParticleSpawnWidth(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Num Particles:
                      <input 
                        type="number"
                        value={numParticles}
                        onChange={(e) => setNumParticles(parseInt(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Show Trajectories:
                      <input 
                        type="checkbox"
                        checked={showTrajectories}
                        onChange={(e) => setShowTrajectories(e.target.checked)}
                        className={`ml-2 ${isDark ? 'bg-gray-700 border-gray-600' : 'bg-gray-100 border-gray-300'}`}
                      />
                    </label>
                  </div>
                  <div className="space-y-2">
                    <button 
                      onClick={handleRunSimulation}
                      className={`w-full px-3 py-2 rounded font-medium transition-colors ${
                        isSimulating 
                          ? 'bg-gray-400 cursor-not-allowed' 
                          : isDark
                            ? 'bg-green-600 hover:bg-green-700 active:bg-green-800'
                            : 'bg-green-500 hover:bg-green-600 active:bg-green-700'
                      } text-white`}
                      disabled={isSimulating}
                    >
                      {isSimulating ? 'Simulating...' : 'Run Simulation'}
                    </button>

                    {/* Simple loading indicator */}
                    <div className={`transition-all duration-300 ${
                      isSimulating ? 'opacity-100 h-8' : 'opacity-0 h-0'
                    }`}>
                      <div className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'} text-center`}>
                        Computing quantum dynamics...
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="hidden lg:block">
                  <h2 className={`text-xl font-bold mb-2 border-t pt-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Potential</h2>
                  <label className={`block mb-2 ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                    <span className={`font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>Type:</span>
                    <select 
                      value={barrierType} 
                      onChange={(e) => setBarrierType(e.target.value)}
                      className={`ml-2 border p-1 rounded ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                    >
                      <option value="delta">Delta Barrier</option>
                      <option value="gaussian">Gaussian Barrier</option>
                      <option value="doubleGaussian">Double Gaussian Barrier</option>
                      <option value="squareWell">Square Well</option>
                    </select>
                  </label>
                  {barrierType === 'delta' && (
                    <div className="space-y-2">
                      <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        Alpha:
                        <input 
                          type="number" 
                          value={deltaAlpha} 
                          onChange={(e) => setDeltaAlpha(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        />
                      </label>
                      <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={deltaZ0} 
                          onChange={(e) => setDeltaZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'squareWell' && (
                    <div className="space-y-2">
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        V0:
                        <input 
                          type="number" 
                          value={gaussV0} 
                          onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        barrier width:
                        <input 
                          type="number" 
                          value={gaussSigma} 
                          onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          step="0.1"
                          min="0"
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={gaussZ0} 
                          onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'gaussian' && (
                    <div className="space-y-2">
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        V0:
                        <input 
                          type="number" 
                          value={gaussV0} 
                          onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        sigma:
                        <input 
                          type="number" 
                          value={gaussSigma} 
                          onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          step="0.1"
                          min="-999999"
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={gaussZ0} 
                          onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'doubleGaussian' && (
                    <div className="space-y-4">
                      <div>
                        <h3 className="font-bold dark:text-white">Gaussian 1</h3>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          V0:
                          <input 
                            type="number" 
                            value={gaussV0} 
                            onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma:
                          <input 
                            type="number" 
                            value={gaussSigma} 
                            onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                            step="0.1"
                            min="-999999"
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0:
                          <input 
                            type="number" 
                            value={gaussZ0} 
                            onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          />
                        </label>
                      </div>
                      <div>
                        <h3 className="font-bold dark:text-white">Gaussian 2</h3>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          V0:
                          <input 
                            type="number" 
                            value={gauss2V0} 
                            onChange={(e) => setGauss2V0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma:
                          <input 
                            type="number" 
                            value={gauss2Sigma} 
                            onChange={(e) => setGauss2Sigma(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                            step="0.1"
                            min="-999999"
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0:
                          <input 
                            type="number" 
                            value={gauss2Z0} 
                            onChange={(e) => setGauss2Z0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                      </div>
                    </div>
                  )}
                </div>
                {/* Wave Packet Controls */}
                <div className="hidden lg:block">
                  <h2 className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Wave Packet Controls</h2>
                  <div className="space-y-2">
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      First Wave Packet z0:
                      <input 
                        type="number"
                        value={z0Packet}
                        onChange={(e) => setZ0Packet(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      First Wave Packet p0:
                      <input 
                        type="number"
                        value={p0Packet}
                        onChange={(e) => setP0Packet(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Use Superposition:
                      <input 
                        type="checkbox"
                        checked={useSuperposition}
                        onChange={(e) => setUseSuperposition(e.target.checked)}
                        className={`ml-2 ${isDark ? 'bg-gray-700 border-gray-600' : 'bg-gray-100 border-gray-300'}`}
                      />
                    </label>
                    {useSuperposition && (
                      <div className="ml-4 space-y-2">
                        <h3 className={`font-bold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>Second Wave Packet</h3>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0 (2nd):
                          <input 
                            type="number"
                            value={z0Packet2}
                            onChange={(e) => setZ0Packet2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          p0 (2nd):
                          <input 
                            type="number"
                            value={p0Packet2}
                            onChange={(e) => setP0Packet2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma (2nd):
                          <input 
                            type="number"
                            value={sigmaPacket2}
                            onChange={(e) => setSigmaPacket2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                      </div>
                    )}
                  </div>
                </div>
                {/* Save/Load Section */}
                <div className="border-t pt-4">
                  <details className="cursor-pointer">
                    <summary className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Save & Load</summary>
                    <div className="space-y-2 mt-2">
                      <input 
                        type="text"
                        value={simName}
                        onChange={(e) => setSimName(e.target.value)}
                        className={`border p-1 rounded w-full ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        placeholder="Simulation name"
                      />
                      <button 
                        onClick={handleSaveSimulation}
                        className={`bg-blue-500 text-white px-3 py-1 rounded w-full ${isDark ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-800 hover:bg-gray-700'}`}
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
                                className={`bg-green-400 text-white px-2 py-1 rounded ${isDark ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-800 hover:bg-gray-700'}`}
                              >
                                Load
                              </button>
                              <button
                                onClick={() => handleDeleteSimulation(s.timestamp)}
                                className={`bg-red-400 text-white px-2 py-1 rounded ${isDark ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-800 hover:bg-gray-700'}`}
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

            {/* Sticky footer with logos */}
            <div className="fixed bottom-4 left-4 flex flex-row items-center gap-8 w-80">
              <img 
                src="./assets/uhasselt-liggend.png" 
                alt="UHasselt Logo" 
                className="w-1/3 object-contain bg-white p-2 m-2"
              />
              <img 
                src="./assets/KU_Leuven_logo.png" 
                alt="KU Leuven Logo" 
                className="w-1/3 object-contain"
              />
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 overflow-y-auto">
          {/* Right Sidebar Content (moved to top on mobile) */}
          <div className={`lg:hidden p-4 border-b ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
            <h2 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Simulation Setup</h2>
            <div className="visible lg:hidden">
                  <h2 className={`text-xl font-bold mb-2 border-t pt-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Potential</h2>
                  <label className={`block mb-2 ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                    <span className={`font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>Type:</span>
                    <select 
                      value={barrierType} 
                      onChange={(e) => setBarrierType(e.target.value)}
                      className={`ml-2 border p-1 rounded ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                    >
                      <option value="delta">Delta Barrier</option>
                      <option value="gaussian">Gaussian Barrier</option>
                      <option value="doubleGaussian">Double Gaussian Barrier</option>
                      <option value="squareWell">Square Well</option>
                    </select>
                  </label>
                  {barrierType === 'delta' && (
                    <div className="space-y-2">
                      <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        Alpha:
                        <input 
                          type="number" 
                          value={deltaAlpha} 
                          onChange={(e) => setDeltaAlpha(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        />
                      </label>
                      <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={deltaZ0} 
                          onChange={(e) => setDeltaZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'squareWell' && (
                    <div className="space-y-2">
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        V0:
                        <input 
                          type="number" 
                          value={gaussV0} 
                          onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        barrier width:
                        <input 
                          type="number" 
                          value={gaussSigma} 
                          onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          step="0.1"
                          min="0"
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={gaussZ0} 
                          onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'gaussian' && (
                    <div className="space-y-2">
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        V0:
                        <input 
                          type="number" 
                          value={gaussV0} 
                          onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        sigma:
                        <input 
                          type="number" 
                          value={gaussSigma} 
                          onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          step="0.1"
                          min="-999999"
                        />
                      </label>
                      <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                        z0:
                        <input 
                          type="number" 
                          value={gaussZ0} 
                          onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                          className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                        />
                      </label>
                    </div>
                  )}
                  {barrierType === 'doubleGaussian' && (
                    <div className="space-y-4">
                      <div>
                        <h3 className="font-bold dark:text-white">Gaussian 1</h3>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          V0:
                          <input 
                            type="number" 
                            value={gaussV0} 
                            onChange={(e) => setGaussV0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma:
                          <input 
                            type="number" 
                            value={gaussSigma} 
                            onChange={(e) => setGaussSigma(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                            step="0.1"
                            min="-999999"
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0:
                          <input 
                            type="number" 
                            value={gaussZ0} 
                            onChange={(e) => setGaussZ0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
                          />
                        </label>
                      </div>
                      <div>
                        <h3 className="font-bold dark:text-white">Gaussian 2</h3>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          V0:
                          <input 
                            type="number" 
                            value={gauss2V0} 
                            onChange={(e) => setGauss2V0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma:
                          <input 
                            type="number" 
                            value={gauss2Sigma} 
                            onChange={(e) => setGauss2Sigma(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                            step="0.1"
                            min="-999999"
                          />
                        </label>
                        <label className={`block font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0:
                          <input 
                            type="number" 
                            value={gauss2Z0} 
                            onChange={(e) => setGauss2Z0(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                      </div>
                    </div>
                  )}
                </div>
                {/* Wave Packet Controls */}
                <div className="visible lg:hidden">
                  <h2 className={`text-xl font-bold mt-2 mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Wave Packet Controls</h2>
                  <div className="space-y-2">
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      First Wave Packet z0:
                      <input 
                        type="number"
                        value={z0Packet}
                        onChange={(e) => setZ0Packet(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      First Wave Packet p0:
                      <input 
                        type="number"
                        value={p0Packet}
                        onChange={(e) => setP0Packet(parseFloat(e.target.value))}
                        className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                      />
                    </label>
                    <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                      Use Superposition:
                      <input 
                        type="checkbox"
                        checked={useSuperposition}
                        onChange={(e) => setUseSuperposition(e.target.checked)}
                        className={`ml-2 ${isDark ? 'bg-gray-700 border-gray-600' : 'bg-gray-100 border-gray-300'}`}
                      />
                    </label>
                    {useSuperposition && (
                      <div className="ml-4 space-y-2">
                        <h3 className={`font-bold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>Second Wave Packet</h3>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          z0 (2nd):
                          <input 
                            type="number"
                            value={z0Packet2}
                            onChange={(e) => setZ0Packet2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          p0 (2nd):
                          <input 
                            type="number"
                            value={p0Packet2}
                            onChange={(e) => setP0Packet2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                        <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                          sigma (2nd):
                          <input 
                            type="number"
                            value={sigmaPacket2}
                            onChange={(e) => setSigmaPacket2(parseFloat(e.target.value))}
                            className={`ml-2 border p-1 rounded w-20 ${isDark ? 'bg-gray-700 border-gray-600 text-gray-200' : 'bg-gray-100 border-gray-300 text-gray-900'}`}
                          />
                        </label>
                      </div>
                    )}
                  </div>
                </div>
            <div className="space-y-2">
              <button 
                onClick={handleRunSimulation}
                className={`w-full px-3 py-2 rounded font-medium transition-colors ${
                  isSimulating 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : isDark
                      ? 'bg-green-600 hover:bg-green-700 active:bg-green-800'
                      : 'bg-green-500 hover:bg-green-600 active:bg-green-700'
                } text-white`}
                disabled={isSimulating}
              >
                {isSimulating ? 'Simulating...' : 'Run Simulation'}
              </button>

              {/* Simple loading indicator */}
              <div className={`transition-all duration-300 ${
                isSimulating ? 'opacity-100 h-8' : 'opacity-0 h-0'
              }`}>
                <div className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'} text-center`}>
                  Computing quantum dynamics...
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="p-4 space-y-6">
            {/* 1D Probability Density Plot */}
            <div className="flex-1">
              <div className="flex items-center mb-2">
                <h3 className={`text-lg font-semibold ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>Probability Density in 1D</h3>
                <InfoButton onClick={() => setActiveExplanation('probabilityCurrent')} />
              </div>
              <Plot
                data={lineData}
                layout={{
                  ...getPlotTheme(),
                  width: undefined,
                  height: 300,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { 
                    title: 'z',
                    range: [xMin, xMax],
                    ...getPlotTheme().xaxis
                  },
                  yaxis: { 
                    title: '|ψ|²',
                    side: 'left',
                    ...getPlotTheme().yaxis
                  },
                  yaxis2: { 
                    title: 'V(z)',
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false,
                    ...getPlotTheme().yaxis
                  },
                  showlegend: true,
                  legend: { x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' },
                }}
                useResizeHandler
                style={{ width: '100%', height: '300px', borderRadius: '10px', overflow: 'hidden' }}
              />
            </div>

            {/* Time Slider - Moved below 1D plot */}
            <div className={`p-4 ${isDark ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow`}>
              <h3 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Time Control</h3>
              <input 
                className={`w-full ${isDark ? 'bg-gray-700' : 'bg-gray-100'}`}
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
                  className={`${isDark ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white px-3 py-1 rounded`}
                  disabled={!probArr || probArr.length === 0}
                >
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <span className={isDark ? 'text-white' : 'text-gray-900'}>
                  T = {tArray[tIndex] ? tArray[tIndex].toFixed(3) : 0}
                </span>
              </div>
            </div>

            {/* Heatmap */}
            <div className="flex-1">
              <div className="flex items-center mb-2">
                <h3 className={`text-lg font-semibold ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>Probability Density Heatmap</h3>
                <InfoButton onClick={() => setActiveExplanation('probabilityDensity')} />
              </div>
              <Plot
                data={heatmapData}
                layout={{
                  ...getPlotTheme(),
                  width: undefined,
                  height: 400,
                  autosize: true,
                  margin: { t: 30, l: 50, r: 10, b: 40 },
                  xaxis: { title: 'time (s)' },
                  yaxis: { title: 'z', range: [xMin, xMax] },
                }}
                useResizeHandler
                style={{ width: '100%', height: '400px', borderRadius: '10px', overflow: 'hidden'  }}
              />
            </div>

            {/* Probability Current & Bohmian Trajectories */}
            <div className="flex-1 mt-4">
              <div className="flex items-center mb-2">
                <h3 className={`text-lg font-semibold ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>Probability Current & Bohmian Trajectories</h3>
                <InfoButton onClick={() => setActiveExplanation('bohmianTrajectories')} />
              </div>
              <Plot
                data={trajectoryData}
                layout={{
                  ...getPlotTheme(),
                  width: undefined,
                  height: 400,
                  autosize: true,
                  margin: { t: 50, l: 50, r: 50, b: 40 },
                  xaxis: { title: 'time (s)'},
                  yaxis: { title: 'z', range: [xMin, xMax] },
                  showlegend: false,
                }}
                useResizeHandler
                style={{ width: '100%', height: '400px', borderRadius: '10px', overflow: 'hidden'  }}
              />
            </div>

            {/* Arrival Time Distribution from probability current */}
            <div className="flex-1 mt-4">
              <div className="flex items-center mb-2">
                <h3 className={`text-lg font-semibold ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>Arrival Time Distribution</h3>
                <InfoButton onClick={() => setActiveExplanation('arrivalTimeDistribution')} />
              </div>
              
              <Plot
                data={arrivalTimesData}
                layout={{
                  ...getPlotTheme(),
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
                style={{ width: '100%', height: '300px', borderRadius: '10px', overflow: 'hidden'  }}
              />
            </div>
          </div>
        </div>

        {/* Right Sidebar - Hidden on mobile */}
        <div className={`hidden lg:block w-1/5 min-w-[250px] border-l ${isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-300 bg-white'} p-4 overflow-y-auto`}>
          <div className="space-y-4">
            <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>Simulation Controls</h2>
            <div className="space-y-2">
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                xMin:
                <input 
                  type="number"
                  value={xMin}
                  onChange={(e) => setXMin(parseFloat(e.target.value))}
                  className={`ml-2 border p-1 rounded w-16 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                  step="any"
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                xMax:
                <input 
                  type="number"
                  value={xMax}
                  onChange={(e) => setXMax(parseFloat(e.target.value))}
                  className={`ml-2 border p-1 rounded w-16 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Nx:
                <div className="flex items-center gap-2">
                  <input 
                    type="number"
                    value={Nx}
                    onChange={(e) => setNx(parseInt(e.target.value))}
                    className={`ml-2 border p-1 rounded w-16 ${
                      isDark 
                        ? 'bg-gray-700 border-gray-600 text-gray-200' 
                        : 'bg-white border-gray-300'
                    }`}
                  />
                  <button
                    onClick={() => setNx(prev => Math.max(2, prev / 2))}
                    className={`px-2 py-1 text-sm rounded ${
                      isDark 
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                        : 'bg-gray-200 hover:bg-gray-300'
                    }`}
                    title="Halve Nx"
                  >
                    /2
                  </button>
                  <button
                    onClick={() => setNx(prev => prev * 2)}
                    className={`px-2 py-1 text-sm rounded ${
                      isDark 
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                        : 'bg-gray-200 hover:bg-gray-300'
                    }`}
                    title="Double Nx"
                  >
                    *2
                  </button>
                </div>
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                nSteps:
                <input 
                  type="number"
                  value={nSteps}
                  onChange={(e) => setNSteps(parseInt(e.target.value))}
                  className={`ml-2 border p-1 rounded w-16 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                dt:
                <input 
                  type="number"
                  value={dt}
                  onChange={(e) => setDt(parseFloat(e.target.value))}
                  className={`ml-2 border p-1 rounded w-16 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                  step="0.001"
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Detector at -L:
                <input 
                  type="number"
                  value={detectorL}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) setDetectorL(val);
                  }}
                  className={`ml-2 border p-1 rounded w-20 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                  step="any"
                />
              </label>
              {/* Particle Trajectory Parameters */}
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Bohmian Trajectory Spawn Center:
                <input 
                  type="number"
                  value={particleSpawnCenter}
                  onChange={(e) => setParticleSpawnCenter(parseFloat(e.target.value))}
                  className={`ml-2 border p-1 rounded w-20 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Bohmian Trajectory Spawn Width:
                <input 
                  type="number"
                  value={particleSpawnWidth}
                  onChange={(e) => setParticleSpawnWidth(parseFloat(e.target.value))}
                  className={`ml-2 border p-1 rounded w-20 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Num Particles:
                <input 
                  type="number"
                  value={numParticles}
                  onChange={(e) => setNumParticles(parseInt(e.target.value))}
                  className={`ml-2 border p-1 rounded w-20 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-gray-200' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
              <label className={`block ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Show Trajectories:
                <input 
                  type="checkbox"
                  checked={showTrajectories}
                  onChange={(e) => setShowTrajectories(e.target.checked)}
                  className={`ml-2 ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600' 
                      : 'bg-white border-gray-300'
                  }`}
                />
              </label>
            </div>
            <div className="space-y-2">
              <button 
                onClick={handleRunSimulation}
                className={`w-full px-3 py-2 rounded font-medium transition-colors ${
                  isSimulating 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : isDark
                      ? 'bg-green-600 hover:bg-green-700 active:bg-green-800'
                      : 'bg-green-500 hover:bg-green-600 active:bg-green-700'
                } text-white`}
                disabled={isSimulating}
              >
                {isSimulating ? 'Simulating...' : 'Run Simulation'}
              </button>

              {/* Simple loading indicator */}
              <div className={`transition-all duration-300 ${
                isSimulating ? 'opacity-100 h-8' : 'opacity-0 h-0'
              }`}>
                <div className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'} text-center`}>
                  Computing quantum dynamics...
                </div>
              </div>
            </div>
            <div className="mt-4">
              <h3 className={`border-t ${isDark ? 'border-gray-700' : 'border-gray-200'} pt-2 font-semibold ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Wavefunction Distribution:
              </h3>
              <p className={`${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                x &lt; 0 (Transmitted):{" "}
                {(() => {
                  if (!probArr || zArray.length < 2 || !cumulativeLossArrLeft || !cumulativeLossArrRight) return 0;
                  const slice = probArr[tIndex] || [];
                  const dz = zArray[1] - zArray[0];
                  let leftSum = 0,
                      rightSum = 0;
                  for (let i = 0; i < zArray.length; i++) {
                    if (zArray[i] < 0) {
                      leftSum += slice[i] * dz;
                    } else {
                      rightSum += slice[i] * dz;
                    }
                  }
                  const leftLoss = cumulativeLossArrLeft[tIndex] || 0;
                  const rightLoss = cumulativeLossArrRight[tIndex] || 0;
                  const total = leftSum + rightSum + leftLoss + rightLoss;
                  return total ? ((leftSum + leftLoss) / total * 100).toFixed(2) : 0;
                })()}
                %
              </p>
              <p className={`${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                x ≥ 0 (Reflected):{" "}
                {(() => {
                  if (!probArr || zArray.length < 2 || !cumulativeLossArrLeft || !cumulativeLossArrRight) return 0;
                  const slice = probArr[tIndex] || [];
                  const dz = zArray[1] - zArray[0];
                  let leftSum = 0,
                      rightSum = 0;
                  for (let i = 0; i < zArray.length; i++) {
                    if (zArray[i] < 0) {
                      leftSum += slice[i] * dz;
                    } else {
                      rightSum += slice[i] * dz;
                    }
                  }
                  const leftLoss = cumulativeLossArrLeft[tIndex] || 0;
                  const rightLoss = cumulativeLossArrRight[tIndex] || 0;
                  const total = leftSum + rightSum + leftLoss + rightLoss;
                  return total ? ((rightSum + rightLoss) / total * 100).toFixed(2) : 0;
                })()}
                %
              </p>
              <p className={`${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Left Loss:{" "}
                {(() => {
                  if (!probArr || zArray.length < 2 || !cumulativeLossArrLeft || !cumulativeLossArrRight) return 0;
                  const slice = probArr[tIndex] || [];
                  const dz = zArray[1] - zArray[0];
                  let leftSum = 0,
                      rightSum = 0;
                  for (let i = 0; i < zArray.length; i++) {
                    if (zArray[i] < 0) {
                      leftSum += slice[i] * dz;
                    } else {
                      rightSum += slice[i] * dz;
                    }
                  }
                  const leftLoss = cumulativeLossArrLeft[tIndex] || 0;
                  const rightLoss = cumulativeLossArrRight[tIndex] || 0;
                  const total = leftSum + rightSum + leftLoss + rightLoss;
                  return total ? (leftLoss / total * 100).toFixed(2) : 0;
                })()}
                %
              </p>
              <p className={`${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Right Loss:{" "}
                {(() => {
                  if (!probArr || zArray.length < 2 || !cumulativeLossArrLeft || !cumulativeLossArrRight) return 0;
                  const slice = probArr[tIndex] || [];
                  const dz = zArray[1] - zArray[0];
                  let leftSum = 0,
                      rightSum = 0;
                  for (let i = 0; i < zArray.length; i++) {
                    if (zArray[i] < 0) {
                      leftSum += slice[i] * dz;
                    } else {
                      rightSum += slice[i] * dz;
                    }
                  }
                  const leftLoss = cumulativeLossArrLeft[tIndex] || 0;
                  const rightLoss = cumulativeLossArrRight[tIndex] || 0;
                  const total = leftSum + rightSum + leftLoss + rightLoss;
                  return total ? (rightLoss / total * 100).toFixed(2) : 0;
                })()}
                %
              </p>
              <p className={`${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                Total Loss:{" "}
                {(() => {
                  if (!probArr || zArray.length < 2 || !cumulativeLossArrLeft || !cumulativeLossArrRight) return 0;
                  const slice = probArr[tIndex] || [];
                  const dz = zArray[1] - zArray[0];
                  let leftSum = 0,
                      rightSum = 0;
                  for (let i = 0; i < zArray.length; i++) {
                    if (zArray[i] < 0) {
                      leftSum += slice[i] * dz;
                    } else {
                      rightSum += slice[i] * dz;
                    }
                  }
                  const leftLoss = cumulativeLossArrLeft[tIndex] || 0;
                  const rightLoss = cumulativeLossArrRight[tIndex] || 0;
                  const total = leftSum + rightSum + leftLoss + rightLoss;
                  return total ? ((leftLoss + rightLoss) / total * 100).toFixed(2) : 0;
                })()}
                %
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Modals */}
      <InfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} isDark={isDark} />
      <ExplanationModal
        isOpen={!!activeExplanation}
        onClose={() => setActiveExplanation(null)}
        content={activeExplanation ? explanations[activeExplanation] : ''}
        isDark={isDark}
      />

      {/* Mobile menu overlay */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
    </div>
  );
};

// Wrap the Dashboard with ThemeProvider
const DashboardWithTheme = () => (
  <ThemeProvider>
    <Dashboard />
  </ThemeProvider>
);

export default DashboardWithTheme;
