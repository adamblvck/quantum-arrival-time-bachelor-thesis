// 2D Plotting Helpers for Quantum Simulations

/**
 * Generates a 3D surface plot data for the probability density |ψ(x,y)|².
 * For use with Plotly's 'surface' type.
 */
export const generate3DSurfaceData = (prob2D, xArr, yArr) => {
  return [{
    x: xArr,
    y: yArr,
    z: prob2D,  // prob2D should be a 2D array [Nx x Ny]
    type: 'surface',
    colorscale: 'Viridis',
    name: '|ψ(x,y)|²'
  }];
};

/**
 * (Optional) Generates 2D quiver (or vector) data for the probability current.
 * Here one would need to compute the gradients (using finite differences) to get (jx, jy).
 * One can then plot using scatter markers with arrow annotations.
 */
export const generate2DCurrentData = (currentX, currentY, xArr, yArr) => {
  // For illustration, assume currentX and currentY are 2D arrays of the same size as the grid.
  const data = [];
  for (let i = 0; i < xArr.length; i++) {
    for (let j = 0; j < yArr.length; j++) {
      data.push({
        x: [xArr[i]],
        y: [yArr[j]],
        u: [currentX[i][j]],  // arrow in x direction
        v: [currentY[i][j]]   // arrow in y direction
      });
    }
  }
  // Actual rendering might require a custom quiver plot implementation or annotations.
  return data;
};

/**
 * Generates data for displaying Bohmian trajectories in 2D.
 * Each trajectory is an array of { x, y } positions over time.
 */
export const generateTrajectoryData2D = (trajectories) => {
  // trajectories: an array of trajectories, each trajectory being an array of { x, y }.
  return trajectories.map(traj => ({
    x: traj.map(pos => pos.x),
    y: traj.map(pos => pos.y),
    mode: 'lines',
    line: { width: 1, color: 'black' },
    showlegend: false
  }));
};
