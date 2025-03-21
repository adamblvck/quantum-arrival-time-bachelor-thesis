import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { generate3DSurfaceData } from '../physics/plotting2D';

const Dashboard2DSimulation = () => {
  // Simulation parameters (editable via UI if desired)
  const [xMin, setXMin] = useState(-20);
  const [xMax, setXMax] = useState(20);
  const [Nx, setNx] = useState(128);
  const [yMin, setYMin] = useState(-20);
  const [yMax, setYMax] = useState(20);
  const [Ny, setNy] = useState(512);
  const [nSteps, setNSteps] = useState(1000);
  const [dt, setDt] = useState(0.01);
  const [hbar, setHbar] = useState(1.0);
  const [m, setM] = useState(1.0);
  
  // Potential & initial packet parameters
  const [barrierCenterX, setBarrierCenterX] = useState(0.0);
  const [barrierStrength, setBarrierStrength] = useState(5.0);
  const [barrierWidth, setBarrierWidth] = useState(0.5);
  const [x0, setX0] = useState(-5.0);
  const [y0, setY0] = useState(-5.0);
  const [p0x, setP0x] = useState(0);
  const [p0y, setP0y] = useState(0);
  const [sigma, setSigma] = useState(1.0);

  // State for simulation results and time slider
  const [simResult, setSimResult] = useState(null);
  const [tIndex, setTIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  const handleRunSimulation = async () => {
    setLoading(true);
    // Construct query parameters for the API call
    const params = new URLSearchParams({
      x_min: xMin,
      x_max: xMax,
      Nx: Nx,
      y_min: yMin,
      y_max: yMax,
      Ny: Ny,
      n_steps: nSteps,
      dt: dt,
      hbar: hbar,
      m: m,
      barrier_center_x: barrierCenterX,
      barrier_strength: barrierStrength,
      barrier_width: barrierWidth,
      x0: x0,
      y0: y0,
      p0x: p0x,
      p0y: p0y,
      sigma: sigma
    });

    try {
      const response = await fetch(`http://127.0.0.1:5000/simulate?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setSimResult(data);
      setTIndex(0);
    } catch (error) {
      console.error("Simulation error:", error);
      alert(`Failed to run simulation: ${error.message}`);  // Show error to user
    } finally {
      setLoading(false);
    }
  };

  // Generate the 3D surface data from the probability density at the current time index
  const surfaceData = simResult ? generate3DSurfaceData(simResult.prob[tIndex], simResult.x, simResult.y) : [];

  return (
    <div className="dashboard-2d">
      <h2>2D Quantum Simulation Dashboard</h2>
      <div style={{ marginBottom: '1rem' }}>
        <button onClick={handleRunSimulation} disabled={loading}>
          {loading ? "Running Simulation..." : "Run 2D Simulation"}
        </button>
      </div>
      {simResult && (
        <div>
          <Plot
            data={surfaceData}
            layout={{
              title: 'Probability Density |ψ(x,y)|²',
              autosize: true,
              scene: {
                xaxis: { title: 'x' },
                yaxis: { title: 'y' },
                zaxis: { title: '|ψ(x,y)|²' }
              },
              margin: { l: 50, r: 50, b: 50, t: 50 }
            }}
            useResizeHandler
            style={{ width: '100%', height: '500px' }}
          />
          <div style={{ marginTop: '1rem' }}>
            <label>
              Time Step Index: {tIndex}
              <input
                type="range"
                min="0"
                max={simResult.t.length - 1}
                value={tIndex}
                onChange={(e) => setTIndex(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard2DSimulation;
