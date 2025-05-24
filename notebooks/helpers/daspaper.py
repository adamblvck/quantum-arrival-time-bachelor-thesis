import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

def psihat_2(E, sigma, z0):
    """
    Returns the Airy-transform wavefunction hat{psi}(E) from Siddhant Das' Appendix:

        hat{psi}(E) = sqrt(2*sigma*sqrt(pi)) *
                      exp[ 0.5 * sigma^2 * (sigma^4/6 + z0 - E) ] *
                      Ai( sigma^4/4 + z0 - E )

    Parameters
    ----------
    E : float or np.ndarray
        The energy argument (dimensionless or in chosen units).
    sigma : float
        Width parameter from the Gaussian initial wavefunction.
    z0 : float
        The initial center position (in the same units as E).

    Returns
    -------
    np.ndarray
        Value(s) of hat{psi}(E).
    """
    # Extract Ai from airy(...) = (Ai, Ai', Bi, Bi') -> [0] for Airy function
    
    prefactor = np.sqrt(2.0 * sigma * np.sqrt(np.pi))
    exponent = np.exp(0.5 * sigma**2 * ( (sigma**4)/6 + z0 - E))
    Ai_val = airy((sigma**4)/4 + z0 - E)[0]
    return prefactor * exponent * Ai_val

def psihat_2_corrected(E, sigma, z0):
    """
    Returns the Airy-transform wavefunction hat{psi}(E) from Siddhant Das' Appendix:

	Adam Corrected

    Parameters
    ----------
    E : float or np.ndarray
        The energy argument (dimensionless or in chosen units).
    sigma : float
        Width parameter from the Gaussian initial wavefunction.
    z0 : float
        The initial center position (in the same units as E).

    Returns
    -------
    np.ndarray
        Value(s) of hat{psi}(E).
    """
    # Extract Ai from airy(...) = (Ai, Ai', Bi, Bi') -> [0] for Airy function
    
    prefactor = (8*np.pi*sigma**2)**(1/4)
    exponent = np.exp( 2*sigma**6/3 + (sigma**2) * (z0 - E) )
    Ai_val = airy((sigma**4)/4 + z0 - E)[0]
    return prefactor * exponent * Ai_val


def P_gamma_of_tau_2(tau, sigma, z0, L, gamma):
    """
    Computes:
        P_gamma(tau) = [2 * tau * |psihat(L + tau^2)|^2]
                       / [1 + (2*pi*gamma * Ai^2(-L - tau^2))^2]

    Parameters
    ----------
    tau : float
        The variable with respect to which we plot P_gamma(tau).
    sigma : float
        Wavepacket width parameter from the Gaussian initial condition.
    z0 : float
        Initial center position in the transform argument.
    L : float
        The shift in the argument (e.g., from the problem statement).
    gamma : float
        Dimensionless parameter multiplying Ai^2 in the denominator.

    Returns
    -------
    float
        The value of P_gamma(tau).
    """
    # Evaluate hat{psi}(E = L + tau^2)
    psihat_val = psihat_2(-L + tau**2, sigma, z0)
    # psihat_val = psihat_2_corrected(-L + tau**2, sigma, z0)
    psihat_sq = np.abs(psihat_val)**2  # magnitude squared

    # Numerator: 2 * tau * |psihat|^2
    numerator = 2.0 * tau * psihat_sq

    # Denominator: 1 + [2*pi*gamma * Ai^2(-L - tau^2)]^2
    Ai_neg = airy(- L - tau**2)[0]  # Ai(-L - tau^2)
    denominator = 1.0 + (2.0 * np.pi * gamma * Ai_neg**2)**2

    return numerator / denominator

def plot_Py_gamma_2(sigma=1.0, z0=0.0, L=1.0, gamma=0.1,
                  tau_min=0.0, tau_max=5.0, num_points=600):
    """
    Plots the function:
        P_gamma(tau) = 2 tau |psihat(L + tau^2)|^2
                       / [1 + (2 pi gamma Ai^2(-L - tau^2))^2]
    using the hat{psi}(E) given in your Appendix.

    Parameters
    ----------
    sigma : float
        Gaussian wavepacket width parameter.
    z0 : float
        Initial center position of Airy transformed wavepacket
    L : float
        Detector location at -L
    gamma : float
        Dimensionless parameter - gamma * delta
    tau_min : float
        Lower limit of tau for plotting.
    tau_max : float
        Upper limit of tau for plotting.
    num_points : int
        Number of tau values in the space.

    Returns
    -------
    None
        Displays a matplotlib plot of P_gamma(tau).
    """
    tau_vals = np.linspace(tau_min, tau_max, num_points)
    P_vals = [P_gamma_of_tau_2(t, sigma, z0, L, gamma) for t in tau_vals]

    plt.figure(figsize=(7,5))
    plt.plot(tau_vals, P_vals, 'b-', label=r'$P_\gamma(\tau)$')
    plt.xlabel(r'$\tau$', fontsize=12)
    plt.ylabel(r'$P_\gamma(\tau)$', fontsize=12)
    plt.title('Plot of $P_\\gamma(\\tau)$', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()