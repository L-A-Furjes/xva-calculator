"""
Correlation handling with Cholesky decomposition for multi-asset simulation.

Provides utilities for generating correlated random numbers used in
joint simulation of domestic rates, foreign rates, and FX.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray


@dataclass
class CholeskyCorrelation:
    """
    Manages correlation structure for multi-asset Monte Carlo simulation.

    Uses Cholesky decomposition to transform independent standard normal
    random variables into correlated ones.

    For a 3×3 correlation matrix with assets [domestic, foreign, FX]:
        L = cholesky(Σ)
        Z_correlated = L @ Z_independent

    Attributes
    ----------
    rho_df : float
        Correlation between domestic and foreign rates
    rho_dx : float
        Correlation between domestic rate and FX
    rho_fx : float
        Correlation between foreign rate and FX

    Example
    -------
    >>> corr = CholeskyCorrelation(rho_df=0.7, rho_dx=-0.3, rho_fx=0.4)
    >>> z_ind = np.random.standard_normal((1000, 3))
    >>> z_corr = corr.correlate(z_ind)
    >>> print(f"Empirical corr(d,f): {np.corrcoef(z_corr[:,0], z_corr[:,1])[0,1]:.2f}")
    """

    rho_df: float = 0.7  # domestic-foreign
    rho_dx: float = -0.3  # domestic-FX
    rho_fx: float = 0.4  # foreign-FX

    def __post_init__(self) -> None:
        """Validate correlations and compute Cholesky factor."""
        # Validate individual correlations
        for name, val in [
            ("rho_df", self.rho_df),
            ("rho_dx", self.rho_dx),
            ("rho_fx", self.rho_fx),
        ]:
            if not -1 <= val <= 1:
                raise ValueError(f"{name} must be in [-1, 1], got {val}")

        # Build and validate correlation matrix
        self._corr_matrix = self._build_correlation_matrix()
        self._validate_positive_definite()
        self._cholesky = np.linalg.cholesky(self._corr_matrix)

    def _build_correlation_matrix(self) -> FloatArray:
        """Build 3×3 correlation matrix."""
        return np.array(
            [
                [1.0, self.rho_df, self.rho_dx],
                [self.rho_df, 1.0, self.rho_fx],
                [self.rho_dx, self.rho_fx, 1.0],
            ]
        )

    def _validate_positive_definite(self) -> None:
        """Check that correlation matrix is positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self._corr_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Correlation matrix is not positive semi-definite. "
                f"Eigenvalues: {eigenvalues}. "
                f"Check that correlations are consistent."
            )

    @property
    def correlation_matrix(self) -> FloatArray:
        """Return the 3×3 correlation matrix."""
        return self._corr_matrix.copy()

    @property
    def cholesky_factor(self) -> FloatArray:
        """Return the lower triangular Cholesky factor."""
        return self._cholesky.copy()

    def correlate(self, z_independent: FloatArray) -> FloatArray:
        """
        Transform independent normals to correlated normals.

        Parameters
        ----------
        z_independent : FloatArray
            Array of shape (n_samples, 3) with independent N(0,1) variables
            Columns: [z_domestic, z_foreign, z_fx]

        Returns
        -------
        FloatArray
            Correlated normal variables with the specified correlation structure

        Example
        -------
        >>> corr = CholeskyCorrelation(rho_df=0.8, rho_dx=0.0, rho_fx=0.0)
        >>> z_ind = np.random.standard_normal((10000, 3))
        >>> z_corr = corr.correlate(z_ind)
        >>> # Verify correlation
        >>> emp_corr = np.corrcoef(z_corr[:, 0], z_corr[:, 1])[0, 1]
        >>> print(f"Target: 0.80, Empirical: {emp_corr:.2f}")
        """
        if z_independent.shape[1] != 3:
            raise ValueError(
                f"Expected 3 columns for [domestic, foreign, FX], "
                f"got {z_independent.shape[1]}"
            )

        # Z_correlated = L @ Z_independent.T
        # For efficiency with many samples, we compute row-wise
        return z_independent @ self._cholesky.T

    def generate_correlated_samples(
        self,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Generate correlated random samples for simulation.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps (excluding t=0)
        seed : int | None
            Random seed

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray]
            Three arrays of shape (n_paths, n_steps):
            (z_domestic, z_foreign, z_fx)

        Example
        -------
        >>> corr = CholeskyCorrelation()
        >>> z_d, z_f, z_x = corr.generate_correlated_samples(5000, 20, seed=42)
        >>> print(z_d.shape)  # (5000, 20)
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate independent samples
        z_ind = np.random.standard_normal((n_paths * n_steps, 3))

        # Correlate
        z_corr = self.correlate(z_ind)

        # Reshape to (n_paths, n_steps, 3) then extract
        z_corr = z_corr.reshape(n_paths, n_steps, 3)

        z_domestic = z_corr[:, :, 0]
        z_foreign = z_corr[:, :, 1]
        z_fx = z_corr[:, :, 2]

        return z_domestic, z_foreign, z_fx

    @classmethod
    def from_config(
        cls, config: "CorrelationConfig"  # noqa: F821
    ) -> "CholeskyCorrelation":  # noqa: F821
        """
        Create from configuration object.

        Parameters
        ----------
        config : CorrelationConfig
            Configuration with correlation parameters

        Returns
        -------
        CholeskyCorrelation
            Initialized correlation handler
        """
        return cls(
            rho_df=config.domestic_foreign,
            rho_dx=config.domestic_fx,
            rho_fx=config.foreign_fx,
        )
