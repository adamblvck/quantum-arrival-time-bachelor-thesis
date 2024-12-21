import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import { create, all } from 'mathjs';
import { debounce } from 'lodash';
import InfoModal from './InfoModal';

const math = create(all, {});

const Dashboard = () => {
  // User inputs
  const [expression, setExpression] = useState("exp(-(x^2 + y^2))");
  const [expressionValid, setExpressionValid] = useState(true);
  const [functionEnabled, setFunctionEnabled] = useState(true);
  
  // Parameters for normals calculation
  const [a, setA] = useState(Math.PI / 4);
  const [b, setB] = useState(0);
  const [k, setK] = useState(0.1);

  // Flags for toggles
  const [showFx, setShowFx] = useState(true);
  const [showFy, setShowFy] = useState(true);

  // View mode: 'surfaces', 'contour', or 'normals'
  const [viewMode, setViewMode] = useState('surfaces');

  // Animation states
  const [animatingA, setAnimatingA] = useState(false);
  const [animatingB, setAnimatingB] = useState(false);
  const [animatingK, setAnimatingK] = useState(false);
  const animRef = useRef();

  // Grid settings
  const xMin = -5, xMax = 5;
  const yMin = -5, yMax = 5;
  const [numUnits, setNumUnits] = useState(50);
  const xValues = math.range(xMin, xMax, (xMax - xMin) / numUnits).toArray();
  const yValues = math.range(yMin, yMax, (yMax - yMin) / numUnits).toArray();

  // Computed data for the primary chart(s)
  const [plotData, setPlotData] = useState([]);

  // For user input with debounce
  const [tempExpression, setTempExpression] = useState(expression);

  // New state for log scale toggle
  const [logScale, setLogScale] = useState(false);

  // New state for z-axis range
  const [zRange, setZRange] = useState([-10, 10]);

  // New state for min and max values of a and b
  const [aMin, setAMin] = useState(0);
  const [aMax, setAMax] = useState(1000);
  const [bMin, setBMin] = useState(-1);
  const [bMax, setBMax] = useState(1);

  // New state for camera
  const [camera, setCamera] = useState({ eye: { x: 1.5, y: 1.5, z: 1.5 } });

  // ------------ 1. Parse and Compute f, fx, fy --------------

  const computeFunctions = () => {
    try {
      const fExpr = math.parse(expression);
      const fxExpr = math.derivative(fExpr, 'x');
      const fyExpr = math.derivative(fExpr, 'y');

      const f = fExpr.compile();
      const fx = fxExpr.compile();
      const fy = fyExpr.compile();

      setExpressionValid(true);
      return { f, fx, fy };
    } catch (error) {
      console.error("Error parsing expression:", error);
      setExpressionValid(false);
      return { f: null, fx: null, fy: null };
    }
  };

  // ------------ 2. Generate Surfaces for f, fx, fy ----------

  const computeBaseData = (f, fx, fy) => {
    const Zf = [];
    const Zfx = [];
    const Zfy = [];

    for (let yi = 0; yi < yValues.length; yi++) {
      const y = yValues[yi];
      const rowF = [], rowFx = [], rowFy = [];
      for (let xi = 0; xi < xValues.length; xi++) {
        const x = xValues[xi];
        rowF.push(f.evaluate({ x, y }));
        rowFx.push(fx.evaluate({ x, y }));
        rowFy.push(fy.evaluate({ x, y }));
      }
      Zf.push(rowF);
      Zfx.push(rowFx);
      Zfy.push(rowFy);
    }
    return { Zf, Zfx, Zfy };
  };

  // ------------ 3. Distance Between Normals (Surface/Contour) --------

  const computeNormalsDistance = (f, fx, fy) => {
    function normal(x, y) {
      return [fx.evaluate({ x, y }), fy.evaluate({ x, y }), -1.0];
    }

    const Za = [];
    for (let yi = 0; yi < yValues.length; yi++) {
      const y = yValues[yi];
      const rowDist = [];
      for (let xi = 0; xi < xValues.length; xi++) {
        const x = xValues[xi];
        const Pz = f.evaluate({ x, y });
        const P = [x, y, Pz];
        const U = normal(x, y);

        const Qx = x + a;
        const Qy = y + b;
        const Qz = f.evaluate({ x: Qx, y: Qy });
        const Q = [Qx, Qy, Qz];
        const V = normal(Qx, Qy);

        // Cross product of U, V
        const cross = [
          U[1]*V[2] - U[2]*V[1],
          U[2]*V[0] - U[0]*V[2],
          U[0]*V[1] - U[1]*V[0]
        ];
        const crossNorm = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2);

        let dist;
        if (crossNorm < 1e-12) {
          dist = Infinity;  // Lines nearly parallel
        } else {
          const QP = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]];
          const dot = QP[0]*cross[0] + QP[1]*cross[1] + QP[2]*cross[2];
          dist = Math.abs(dot)/crossNorm;
        }

        rowDist.push(dist);
      }
      Za.push(rowDist);
    }
    return Za;
  };

  // ------------ 4. New: Closest-Approach Midpoints (fourth chart) ----

  /**
   * For each point on a coarser grid, solve for s, t to find the closest
   * points on the two normals, then store the midpoint. We'll show these
   * midpoints in a scatter3d trace.
   */
  const computeClosestMidpoints = (f, fx, fy) => {
    // A coarser grid to reduce clutter:
    const coarseStep = 0.1;
    const xVals = math.range(xMin, xMax, coarseStep).toArray();
    const yVals = math.range(yMin, yMax, coarseStep).toArray();

    // Arrays for storing the midpoint coordinates
    const Xmid = [];
    const Ymid = [];
    const Zmid = [];

    // Dot product helper
    const dot = (r1, r2) => r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];

    // We'll define a function for normal vector:
    const normal = (xx, yy) => [
      fx.evaluate({ x:xx, y:yy }),
      fy.evaluate({ x:xx, y:yy }),
      -1
    ];

    for (let yy=0; yy<yVals.length; yy++){
      for (let xx=0; xx<xVals.length; xx++){
        const x0 = xVals[xx];
        const y0 = yVals[yy];

        // Points on the surface
        const Pz = f.evaluate({ x:x0, y:y0 });
        const P = [x0, y0, Pz];

        const Qx = x0 + a;
        const Qy = y0 + b;
        const Qz = f.evaluate({ x:Qx, y:Qy });
        const Q = [Qx, Qy, Qz];

        // Normals
        const U = normal(x0, y0);
        const V = normal(Qx, Qy);

        // Solve for s, t so that (P + sU) - (Q + tV) is perpendicular to U and V
        // (see standard skew-line formula)
        const w = [Q[0]-P[0], Q[1]-P[1], Q[2]-P[2]];

        const A = dot(U,U);
        const B = dot(U,V);
        const C = dot(V,V);
        const alpha = dot(w,U);
        const beta  = dot(w,V);

        const denom = (A*C - B*B);
        if (Math.abs(denom) < 1e-12) {
          // Lines nearly parallel or degenerate => skip
          continue;
        }

        const s = (alpha*C - beta*B)/denom;
        const t = (alpha*B - beta*A)/denom;

        // Closest points
        const Pmin = [P[0] + s*U[0], P[1] + s*U[1], P[2] + s*U[2]];
        const Qmin = [Q[0] + t*V[0], Q[1] + t*V[1], Q[2] + t*V[2]];

        // Midpoint
        const Mx = 0.5*(Pmin[0] + Qmin[0]);
        const My = 0.5*(Pmin[1] + Qmin[1]);
        const Mz = 0.5*(Pmin[2] + Qmin[2]);

        Xmid.push(Mx);
        Ymid.push(My);
        Zmid.push(Mz);
      }
    }

    // Return a Plotly trace for these midpoints
    return {
      type: 'scatter3d',
      mode: 'markers',
      x: Xmid,
      y: Ymid,
      z: Zmid,
      marker: {
        size: 3,
        color: 'red'
      },
      name: 'Closest Midpoints'
    };
  };

  // ------------ 5. Debounce user expression input --------------

  const validateExpression = (expr) => {
    try {
      math.parse(expr).compile();
      setExpressionValid(true);
      setExpression(expr); // Update main expression only if valid
    } catch (error) {
      setExpressionValid(false);
    }
  };

  const handleExpressionChange = (e) => {
    const newExpression = e.target.value;
    setTempExpression(newExpression);
    debouncedValidateExpression(newExpression);
  };

  const debouncedValidateExpression = useRef(debounce(validateExpression, 300)).current;

  // ------------ 6. Build Plot Data Based on viewMode + 4th Chart --------

  const updatePlotData = () => {
    // If invalid expression or function disabled, no plots
    if (!expressionValid || !functionEnabled) {
      setPlotData([]);
      return;
    }

    // Parse & compute
    const { f, fx, fy } = computeFunctions();
    if (!f) {
      setPlotData([]);
      return;
    }

    const { Zf, Zfx, Zfy } = computeBaseData(f, fx, fy);

    if (viewMode === 'surfaces') {
      // Show surfaces of f, fx, fy based on toggles
      const data = [];
      data.push({
        type: 'surface',
        x: xValues,
        y: yValues,
        z: Zf,
        name: 'f(x,y)',
        colorscale: 'Viridis',
        showscale: false
      });
      if (showFx) {
        data.push({
          type: 'surface',
          x: xValues,
          y: yValues,
          z: Zfx,
          name: 'fx(x,y)',
          colorscale: 'Hot',
          showscale: false
        });
      }
      if (showFy) {
        data.push({
          type: 'surface',
          x: xValues,
          y: yValues,
          z: Zfy,
          name: 'fy(x,y)',
          colorscale: 'Portland',
          showscale: false
        });
      }

      // Add midpoint trace to the surfaces view
      const midpointTrace = computeClosestMidpoints(f, fx, fy);
      data.push(midpointTrace);

      setPlotData(data);

    } else if (viewMode === 'contour') {
      // For demonstration, we plot a contour of the normal distance
      const Za = computeNormalsDistance(f, fx, fy);
      const data = [{
        type: 'contour',
        x: xValues,
        y: yValues,
        z: Za,
        colorscale: 'Viridis',
        contours: { showlines: false },
        name: 'Contour of Normals Distance'
      }];
      setPlotData(data);

    } else if (viewMode === 'normals') {
      // Third chart: A surface of the normal distance
      const Za = computeNormalsDistance(f, fx, fy);
      const normalSurface = {
        type: 'surface',
        x: xValues,
        y: yValues,
        z: Za,
        name: 'Distance Between Normals',
        colorscale: 'Viridis',
        showscale: true
      };

      // **Fourth chart**: A scatter3d trace for the closest midpoints
      const midpointTrace = computeClosestMidpoints(f, fx, fy);

      // Combine them in one Plotly figure with 2 traces
      const data = [normalSurface, midpointTrace];
      setPlotData(data);
    }
  };

  // Recompute plot whenever relevant state changes
  useEffect(() => {
    updatePlotData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expression, showFx, showFy, viewMode, a, b, k, zRange]);

  // ------------ 7. Animations for A and K --------------

  useEffect(() => {
    if (animatingA || animatingB || animatingK) {
      const animate = () => {
        animRef.current = requestAnimationFrame(animate);
        if (animatingA) {
          setA(prev => (prev + 0.01) % Math.PI);
        }
        if (animatingB) {
          setB(prev => (prev + 0.01) % 1);
        }
        if (animatingK) {
          setK(prev => (prev + 0.01) % 1);
        }
      };
      animate();
    } else {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    }
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [animatingA, animatingB, animatingK]);

  // ------------ 8. Render UI & Plots --------------

  const [equations, setEquations] = useState(() => {
    const savedEquations = localStorage.getItem('equations');
    return savedEquations ? JSON.parse(savedEquations) : [];
  });

  // Function to save an equation
  const saveEquation = () => {
    const newEquations = [...equations, { id: Date.now(), expression }];
    setEquations(newEquations);
    localStorage.setItem('equations', JSON.stringify(newEquations));
  };

  // Function to delete an equation
  const deleteEquation = (id) => {
    const updatedEquations = equations.filter(eq => eq.id !== id);
    setEquations(updatedEquations);
    localStorage.setItem('equations', JSON.stringify(updatedEquations));
  };

  const [menuOpen, setMenuOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="flex h-screen">
      {/* Information Button */}
      <button 
        onClick={() => setIsModalOpen(true)} 
        className="absolute top-4 left-4 bg-blue-500 text-white p-2 rounded"
      >
        Info
      </button>

      {/* Info Modal */}
      <InfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />

      {/* Left column: View menu */}
      <div className="w-1/5 border-r border-gray-300 p-4 flex flex-col space-y-4">
        <h2 className="text-xl font-bold mb-4">Menu</h2>
        
        {/* Function Expression */}
        <div>
          <label className="block font-semibold mb-2">Function f(x,y):</label>
          <input 
            type="text"
            value={tempExpression}
            onChange={handleExpressionChange}
            className={`border w-full p-2 rounded ${expressionValid ? 'border-green-500' : 'border-red-500'}`}
            placeholder="e.g. sin(x)*cos(y)"
          />
        </div>

        <div className="flex items-center mt-2">
          <input 
            type="checkbox" 
            checked={functionEnabled} 
            onChange={() => setFunctionEnabled(!functionEnabled)} 
            className="mr-2"
          />
          <label className="font-semibold">Enable Function</label>
        </div>

        {/* View Mode */}
        <div>
          <label className="block font-semibold mb-2">View Mode:</label>
          <div className="space-y-1">
            <label className="block">
              <input 
                type="radio" 
                name="view" 
                value="surfaces"
                checked={viewMode==='surfaces'} 
                onChange={() => setViewMode('surfaces')} 
                className="mr-2"
              />
              Surfaces (f, fx, fy)
            </label>
            <label className="block">
              <input 
                type="radio" 
                name="view" 
                value="contour"
                checked={viewMode==='contour'} 
                onChange={() => setViewMode('contour')} 
                className="mr-2"
              />
              Contour
            </label>
            <label className="block">
              <input 
                type="radio" 
                name="view" 
                value="normals"
                checked={viewMode==='normals'} 
                onChange={() => setViewMode('normals')} 
                className="mr-2"
              />
              Normals Distance
            </label>
			<label className="block">
              <input 
                type="radio" 
                name="view" 
                value="normals"
                checked={viewMode==='normals'} 
                onChange={() => setViewMode('normals')} 
                className="mr-2"
              />
              Normals Distance
            </label>
          </div>
        </div>
        
        {/* Show partial derivatives if Surfaces */}
        {viewMode==='surfaces' && (
          <div className="space-y-2">
            <h3 className="font-semibold">Partial Surfaces:</h3>
            <label className="flex items-center">
              <input type="checkbox" checked={showFx} onChange={() => setShowFx(!showFx)} className="mr-2"/>
              Show fx
            </label>
            <label className="flex items-center">
              <input type="checkbox" checked={showFy} onChange={() => setShowFy(!showFy)} className="mr-2"/>
              Show fy
            </label>
          </div>
        )}

        {/* New toggle for log scale */}
        {viewMode === 'normals' && (
          <div className="flex items-center mt-2">
            <input 
              type="checkbox" 
              checked={logScale} 
              onChange={() => setLogScale(!logScale)} 
              className="mr-2"
            />
            <label className="font-semibold">Logarithmic Z-Axis for Midpoints</label>
          </div>
        )}

        {/* New slider for number of units */}
        <div className="space-y-2">
          <h3 className="font-semibold">Grid Units:</h3>
          <label className="block">
            Units: {numUnits}
            <input 
              type="range" 
              min="10" 
              max="100" 
              step="1" 
              value={numUnits} 
              onChange={(e) => setNumUnits(parseInt(e.target.value, 10))} 
              className="w-full"
            />
          </label>
        </div>

        {/* New slider for z-axis range */}
        <div className="space-y-2">
          <h3 className="font-semibold">Z-Axis Limit:</h3>
          <label className="block">
            Min: {zRange[0].toFixed(2)}, Max: {zRange[1].toFixed(2)}
            <input 
              type="range" 
              min="-20" 
              max="20" 
              step="0.1" 
              value={zRange[0]} 
              onChange={(e) => setZRange([parseFloat(e.target.value), zRange[1]])} 
              className="w-full"
            />
            <input 
              type="range" 
              min="-20" 
              max="20" 
              step="0.1" 
              value={zRange[1]} 
              onChange={(e) => setZRange([zRange[0], parseFloat(e.target.value)])} 
              className="w-full"
            />
          </label>
        </div>

        {/* Collapsible menu */}
        <button className="text-lg font-bold" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? 'Hide Menu' : 'Show Menu'}
        </button>
        {menuOpen && (
          <div>
            {/* Equation management */}
            <div className="space-y-2">
              <h3 className="font-semibold">Stored Equations:</h3>
              <button onClick={saveEquation} className="bg-green-500 text-white px-3 py-1 rounded">Save Equation</button>
              <ul>
                {equations.map(eq => (
                  <li key={eq.id} className="flex justify-between">
                    <span>{eq.expression}</span>
                    <button onClick={() => deleteEquation(eq.id)} className="text-red-500">Delete</button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
      
      {/* Middle column: main display */}
      <div className="w-3/5 border-r border-gray-300 p-4 flex flex-col relative">
        <h2 className="text-xl font-bold mb-4">
          {viewMode==='surfaces' && "3D Surfaces"}
          {viewMode==='contour' && "Normals Distance (Contour)"}
          {viewMode==='normals' && "Normals Distance + Midpoints"}
        </h2>
        <div className="flex-1 overflow-auto">
          <Plot
            data={plotData}
            layout={{
              title: '',
              autosize: true,
              scene: { 
                camera: camera,
                xaxis: { type: 'linear', range: [-10, 10] },
                yaxis: { type: 'linear', range: [-10, 10] },
                zaxis: { type: logScale ? 'log' : 'linear', range: zRange }
              },
              margin: { t:50, l:0, r:0, b:0 }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
            config={{ responsive: true }}
            onRelayout={(figure) => {
              if (figure['scene.camera']) {
                setCamera(figure['scene.camera']);
              }
            }}
          />
        </div>
      </div>

      {/* Right column: tools and settings */}
      <div className="w-1/5 p-4 flex flex-col space-y-4">
        <h2 className="text-xl font-bold mb-4">Tools & Settings</h2>
        
        <div className="space-y-2">
          <h3 className="font-semibold">Parameters:</h3>
          <label className="block">
            a: {a.toFixed(2)}
            <input 
              type="range" 
              min={aMin} 
              max={aMax} 
              step="0.01" 
              value={a} 
              onChange={(e) => setA(parseFloat(e.target.value))} 
              className="w-full"
            />
          </label>
          <label className="block">
            b: {b.toFixed(2)}
            <input 
              type="range" 
              min={bMin} 
              max={bMax} 
              step="0.01" 
              value={b} 
              onChange={(e) => setB(parseFloat(e.target.value))} 
              className="w-full"
            />
          </label>
          <label className="block">
            k: {k.toFixed(2)}
            <input 
              type="range" 
              min="0.01" 
              max="1" 
              step="0.01" 
              value={k} 
              onChange={(e) => setK(parseFloat(e.target.value))} 
              className="w-full"
            />
          </label>
        </div>

        <div className="space-y-2">
          <h3 className="font-semibold">Parameter Limits:</h3>
          <label className="block">
            a Min: {aMin.toFixed(2)}, Max: {aMax.toFixed(2)}
            <input 
              type="range" 
              min="0.0001" 
              max={1000} 
              step="0.001" 
              value={aMin} 
              onChange={(e) => setAMin(parseFloat(e.target.value))} 
              className="w-full"
            />
            <input 
              type="range" 
              min="0.000001" 
              max={1000} 
              step="0.001" 
              value={aMax} 
              onChange={(e) => setAMax(parseFloat(e.target.value))} 
              className="w-full"
            />
          </label>
          <label className="block">
            b Min: {bMin.toFixed(2)}, Max: {bMax.toFixed(2)}
            <input 
              type="range" 
              min="-1" 
              max="1" 
              step="0.01" 
              value={bMin} 
              onChange={(e) => setBMin(parseFloat(e.target.value))} 
              className="w-full"
            />
            <input 
              type="range" 
              min="-1" 
              max="1" 
              step="0.01" 
              value={bMax} 
              onChange={(e) => setBMax(parseFloat(e.target.value))} 
              className="w-full"
            />
          </label>
        </div>

        <div className="space-y-2">
          <h3 className="font-semibold">Animations:</h3>
          <div className="flex items-center space-x-2">
            <button 
              className={`px-3 py-1 rounded ${animatingA ? 'bg-red-500 text-white' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
              onClick={() => setAnimatingA(!animatingA)}>
              {animatingA ? 'Stop A' : 'Animate A'}
            </button>
            <button 
              className={`px-3 py-1 rounded ${animatingB ? 'bg-red-500 text-white' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
              onClick={() => setAnimatingB(!animatingB)}>
              {animatingB ? 'Stop B' : 'Animate B'}
            </button>
            <button 
              className={`px-3 py-1 rounded ${animatingK ? 'bg-red-500 text-white' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
              onClick={() => setAnimatingK(!animatingK)}>
              {animatingK ? 'Stop K' : 'Animate K'}
            </button>
          </div>
          <p className="text-sm text-gray-500">Animations cycle through values of A, B, or K.</p>
        </div>

        {viewMode==='normals' && (
          <div className="space-y-2">
            <h3 className="font-semibold">Normals Calculation & 4th Chart:</h3>
            <p className="text-sm text-gray-600">
              Along with the distance surface, we now show a 4th chart as a 
              <code>scatter3d</code> trace of midpoints where two normals come 
              closest. These midpoints are sampled on a coarser grid for clarity.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;