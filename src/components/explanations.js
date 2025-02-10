// explanations.js
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

// Explanations
// https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html


export const explanations = {
  probabilityDensity: `
# Probability Density

The probability density $|\\psi(z,t)|^2$ represents the likelihood of finding the particle at position $z$ and time $t$. It is defined as:

$$
|\\psi(z,t)|^2 = \\psi^*(z,t)\\,\\psi(z,t)
$$

where $\\psi^*(z,t)$ is the complex conjugate of the wavefunction $\\psi(z,t)$.

### Normalization

For a properly normalized wavefunction, the total probability over all space is:

$$
\\int_{-\\infty}^{\\infty} |\\psi(z,t)|^2\\,dz = 1
$$

This ensures that the particle is found somewhere in space with certainty.

**References:**
- Cohen-Tannoudji, C. et al. *Quantum Mechanics* (1977).
  `,

  probabilityCurrent: `
# Probability Density in 1D

Similar to the heatmap, this plot shows the probability density $|\\psi(z,t)|^2$ as a function of position $z$ and time $t$.

Using the slider in the right panel, one can progress the wavefunction in time.

Probability distribution in regions $x < 0$ and $x\\ge 0$ are calculated for analysing the transmission and reflection of the wavepacket, respectively (see right panel).
However, since we're using Split-Operator Method for the numerical simulaiton, the probability distribution is not conserved over time, as the transmitted part gets absorbed eventually..

**References:**
- ...
  `,

  bohmianTrajectories: `
# Bohmian Trajectories

Bohmian mechanics offers an interpretation of quantum mechanics where particles follow definite trajectories guided by the wavefunction.

### Velocity Field

The velocity $v(z,t)$ for the particles is obtained from the probability current:

$$
v(z,t) = \\frac{j(z,t)}{|\\psi(z,t)|^2}
$$

### Equations of Motion

Particles follow trajectories determined by integrating the velocity:

$$
\\frac{dz}{dt} = v(z,t)
$$

This approach yields a deterministic trajectory for each particle, complementing the probabilistic nature of the wavefunction.

The Bohmian trajectories can be disabled in the right panel.

**References:**
- Bohm, D. *A Suggested Interpretation of the Quantum Theory in Terms of "Hidden" Variables* (1952).  
  `,

  potentialEnergy: `
# Analytical Arrival Time Distribution

This plot shows the analytical arrival time distribution for a particle tunneling through a thin potential barrier in free fall, represented by the probability current $j(z,t)$ at the detector position $-L$ as a function of time $t$:

The parameters for the analytical solution don't correspond with simulation parameters yet!

The arrival time distribution per unit area is calculated as:

$$
\\Pi_{\\gamma}(\\tau) = -J_{z}(-L, \\tau) \\approx \\frac{2\\tau \\left| \\hat{\\psi}(L + \\tau^{2}) \\right|^{2}}{1 + (2\\pi\\gamma)^{2} \\operatorname{Ai}^{4}(-L - \\tau^{2})}.
$$

Making the arrival-time density per unit area:

$$
\\Pi_{\\gamma}(\\tau) \\approx \\frac{\\Pi_{0}(\\tau)}{1 + \\left[ 2\\pi\\gamma \\operatorname{Ai}^{2}(-L - \\tau^{2}) \\right]^{2}}
$$

**References:**
- Siddhant Das, *Tunneling through a thin potential barrier in free fall* (2022).
`,

  arrivalTimeDistribution: `
# Simulated Arrival Time Distribution at $-L$

This plot shows the simulated arrival time distribution at the detector position $-L$ as a function of time $t$.

We essentially simulate the evolution of the wavefunction, graphing the probability current $j(-L,t)$ for an adequate range of time.

For now, there's a discrepancy between the analytical and simulated distributions, which we are working on.

**References:**
- ...
  `,

  simulationParameters: `
# Simulation Parameters and Accuracy

This simulation uses a split-operator method to evolve the wavefunction $\\psi(z,t)$ in time.

### Key Parameters

- **Spatial Domain:**  
The simulation is performed over the domain $z \\in [x_{min}, x_{max}]$ with $N_x$ discrete points.
  
- **Time Evolution:**  
The evolution is computed over $nSteps$ time steps with a time increment $dt$.

- **Split-Operator Method:**  
The kinetic term is efficiently handled via fast Fourier transforms (FFT) while the potential term is applied in position space.

- **Bohmian Trajectories:**  
Trajectories are computed using Euler integration with an adjustable integration factor for finer resolution.

### Accuracy Considerations

- **Grid Resolution:**  
Increasing $N_x$ improves spatial accuracy.
  
- **Time Step:**  
A smaller $dt$ enhances temporal accuracy but increases computational load.
  
- **FFT and Integration:**  
The accuracy of the FFT in the kinetic step and the Euler method for trajectory integration both depend on the smoothness of $\\psi(z,t)$ and the potential landscape.
  `,
};

export const InfoButton = ({ onClick }) => (
  <button
    onClick={onClick}
    className="ml-2 p-1 text-sm text-blue-500 hover:text-blue-700 focus:outline-none"
    title="Learn more"
  >
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
    </svg>
  </button>
);
