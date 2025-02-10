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
# Probability Current

The probability current $j(z,t)$ describes the flow of probability in the quantum system. It is calculated as:

$$
j(z,t) = \\frac{\\hbar}{m}\\,\\text{Im}\\left(\\psi^*(z,t)\\,\\frac{\\partial \\psi(z,t)}{\\partial z}\\right)
$$

### Physical Significance

- **Flow of Probability:** A positive $j(z,t)$ indicates that probability is flowing in the positive $z$-direction, while a negative value indicates flow in the opposite direction.
- **Conservation of Probability:** Together with the continuity equation

$$
\\frac{\\partial |\\psi(z,t)|^2}{\\partial t} + \\frac{\\partial j(z,t)}{\\partial z} = 0,
$$

the probability current guarantees that probability is conserved.

**References:**
- Bohm, D. *A Suggested Interpretation of the Quantum Theory in Terms of "Hidden" Variables* (1952).
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

**References:**
- Bohm, D. *A Suggested Interpretation of the Quantum Theory in Terms of "Hidden" Variables* (1952).  
- Holland, P.R. *The Quantum Theory of Motion* (1993).
  `,

  potentialEnergy: `
# Potential Energy Configuration

The simulation models the total potential energy $V(z)$ as a combination of a barrier potential and the gravitational potential.

### Total Potential

$$
V(z) = V_{barrier}(z) + mgz
$$

### Barrier Types

- **Delta Barrier:**

$$
V_{\\delta}(z) = \\alpha\\,\\delta(z-z_0)
$$

- **Gaussian Barrier:**

$$
V_G(z) = V_0\\,\\exp\\left(-\\frac{(z-z_0)^2}{2\\sigma^2}\\right)
$$

- **Double Gaussian:**  
The sum of two Gaussian potentials, each with its own parameters.

This combined potential governs the evolution of the wavefunction according to the Schrödinger equation.

**References:**
- Razavy, M. *Quantum Theory of Tunneling* (2003).
  `,

  arrivalTimeDistribution: `
# Arrival Time Distribution

The arrival time distribution $\\Pi(\\tau)$ can be characterizes the probability that a particle reaches a detector located at $z = -L$ at time $\\tau$ (Siddhant Das paper (2022).

### Definition

Following the Siddhant Das paper (2022), It is defined as the probability current at some detector location $-L$:

$$
\\Pi(\\tau) \\equiv j(-L, \\tau)
$$

### Asymptotic Approximation

For a system with a delta barrier and gravity, an asymptotic expression is given by:

$$
\\Pi_{\\gamma}(\\tau) \\approx \\frac{2\\tau \\left[\\mathrm{Ai}\\bigl( -(L+\\tau^2) \\bigr)\\right]^2}{1 + \\left[2\\tau \\gamma\\,\\mathrm{Ai}'\\bigl( -(L+\\tau^2) \\bigr)\\right]^2}
$$

where:
- $\\mathrm{Ai}$ is the Airy function,
- $\\mathrm{Ai}'$ is its derivative,
- $\\gamma$ represents the strength of the delta barrier, and
- $L$ is the distance from the barrier to the detector.

### Physical Interpretation

- **Multiple Peaks:** The interaction of the wave packet with the barrier leads to both transmitted and reflected components, resulting in multiple peaks in the arrival time distribution.
- **Airy Oscillations:** The gravitational potential introduces characteristic oscillations (Airy oscillations) into the arrival profile.

**References:**
- Muga, J.G., Sala Mayato, R., Egusquiza, I.L. (Eds.), *Time in Quantum Mechanics* (2008).
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
