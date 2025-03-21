import { useState } from 'react';
import Dashboard1DSimulation from './components/Dashboard1DSimulation'
import Dashboard2DSimulation from './components/Dashboard2DSimulation'
import DashboardTransmissionTest from './components/DashboardTransmissionTest'

function App() {
  const [selectedDashboard, setSelectedDashboard] = useState('2dSim');

  return (
    <div className="min-h-screen bg-gray-100 relative">
      {/* Floating dropdown positioned at top-right */}
      <div className="absolute top-4 right-4">
        <select
          className="bg-white px-4 py-2 rounded-lg shadow-md border border-gray-200 cursor-pointer hover:border-gray-300 transition-colors"
          value={selectedDashboard}
          onChange={(e) => setSelectedDashboard(e.target.value)}
        >
          <option value="1dSim">Arrival Time - 1D Sim</option>
          <option value="2dSim">Arrival Time - 2D Sim</option>
          <option value="gravity">Tunneling Time - 1D Spin-1/2 Sim</option>
          <option value="gravity">Tunneling Time - 2D Spin-1/2 Sim</option>
          <option value="transmission">Transmission Test Dashboard</option>
        </select>
      </div>

      {/* Conditional rendering of dashboards */}
      {selectedDashboard === '1dSim' && <Dashboard1DSimulation />}
      {selectedDashboard === '2dSim' && <Dashboard2DSimulation />}
      {selectedDashboard === 'transmission' && <DashboardTransmissionTest />}
    </div>
  );
}

export default App;
