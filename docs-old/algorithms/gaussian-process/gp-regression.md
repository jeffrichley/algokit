---
algorithm_key: "gp-regression"
tags: [gaussian-process, algorithms, gp-regression, bayesian-inference, probabilistic-regression, uncertainty-quantification]
title: "Gaussian Process Regression"
family: "gaussian-process"
---

# Gaussian Process Regression

{{ algorithm_card("gp-regression") }}

!!! abstract "Overview"
    Gaussian Process Regression (GPR) is a powerful non-parametric Bayesian approach to regression that provides not only point predictions but also uncertainty estimates. Unlike traditional regression methods that assume a specific functional form, GPR can model complex, non-linear relationships while naturally quantifying prediction uncertainty.

    The algorithm works by placing a Gaussian Process prior over functions and updating this prior with observed data to obtain a posterior distribution. This posterior provides both mean predictions and variance estimates, making GPR particularly valuable for applications where uncertainty quantification is important, such as active learning, robust optimization, and decision making under uncertainty.

## Mathematical Formulation

!!! math "Gaussian Process Regression"
    A Gaussian Process is defined by its mean function $m(x)$ and covariance function $k(x, x')$:

    $$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

    Given training data $\{(x_i, y_i)\}_{i=1}^n$, the posterior distribution is:

    $$f_* | X, y, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$$

    Where the predictive mean and variance are:

    $$\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y$$

    $$\sigma_*^2 = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*$$

    Here $K$ is the kernel matrix, $k_*$ is the covariance between test and training points, and $\sigma_n^2$ is the noise variance.

!!! success "Key Properties"
    - **Non-parametric**: No fixed model structure assumptions
    - **Probabilistic**: Provides uncertainty estimates for predictions
    - **Flexible**: Can model complex, non-linear relationships
    - **Bayesian**: Incorporates prior knowledge through kernel choice

## Implementation Approaches

=== "Basic GP Regression (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import cholesky, solve_triangular
    from typing import Tuple, Optional, Callable
    import warnings

    class GaussianProcessRegressor:
        """
        Gaussian Process Regression implementation.

        Args:
            kernel: Kernel function for the GP
            alpha: Regularization parameter (noise variance)
            optimizer: Optimization method for hyperparameters
            n_restarts: Number of optimization restarts
        """

        def __init__(self, kernel=None, alpha=1e-6, optimizer='L-BFGS-B', n_restarts=10):
            self.kernel = kernel
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts = n_restarts
            self.X_train_ = None
            self.y_train_ = None
            self.kernel_ = None
            self.L_ = None
            self.alpha_ = None
            self.log_marginal_likelihood_value_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegressor':
            """
            Fit the Gaussian Process regressor.

            Args:
                X: Training features of shape (n_samples, n_features)
                y: Training targets of shape (n_samples,)

            Returns:
                self
            """
            X, y = np.asarray(X), np.asarray(y)

            # Validate inputs
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            if y.ndim != 1:
                raise ValueError("y must be 1-dimensional")
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")

            self.X_train_ = X
            self.y_train_ = y

            # Initialize kernel if not provided
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)

            # Optimize hyperparameters
            self._optimize_hyperparameters()

            # Compute kernel matrix and Cholesky decomposition
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha
            self.L_ = cholesky(K, lower=True)

            # Solve for alpha
            self.alpha_ = solve_triangular(self.L_, self.y_train_, lower=True)
            self.alpha_ = solve_triangular(self.L_.T, self.alpha_, lower=False)

            return self

        def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """
            Predict using the Gaussian Process regressor.

            Args:
                X: Test features of shape (n_samples, n_features)
                return_std: Whether to return standard deviations

            Returns:
                Tuple of (mean, std) where std is None if return_std=False
            """
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)

            # Compute kernel matrices
            K_star = self.kernel_(self.X_train_, X)
            K_star_star = self.kernel_(X)

            # Mean prediction
            mean = K_star.T @ self.alpha_

            if return_std:
                # Variance prediction
                v = solve_triangular(self.L_, K_star, lower=True)
                var = K_star_star - v.T @ v
                var = np.maximum(var, 1e-10)  # Ensure positive variance
                std = np.sqrt(np.diag(var))
                return mean, std
            else:
                return mean, None

        def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
            """
            Sample from the posterior distribution.

            Args:
                X: Test features of shape (n_samples, n_features)
                n_samples: Number of samples to draw
                random_state: Random state for reproducibility

            Returns:
                Samples from the posterior distribution
            """
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before sampling")

            X = np.asarray(X)

            # Get mean and covariance
            mean, std = self.predict(X, return_std=True)

            # Set random state
            if random_state is not None:
                np.random.seed(random_state)

            # Sample from multivariate normal
            samples = np.random.normal(mean, std, (n_samples, len(mean)))

            return samples

        def _optimize_hyperparameters(self):
            """Optimize kernel hyperparameters using marginal likelihood."""
            def objective(theta):
                self.kernel_.theta = theta
                try:
                    return -self._log_marginal_likelihood()
                except np.linalg.LinAlgError:
                    return 1e25

            # Get initial hyperparameters
            theta0 = self.kernel.theta.copy()
            bounds = self.kernel.bounds

            # Optimize with multiple restarts
            best_theta = theta0.copy()
            best_lml = -objective(theta0)

            for _ in range(self.n_restarts):
                # Random restart
                theta0 = np.random.uniform(
                    [bound[0] for bound in bounds],
                    [bound[1] for bound in bounds]
                )

                try:
                    result = minimize(objective, theta0, bounds=bounds, method=self.optimizer)
                    if result.fun < -best_lml:
                        best_theta = result.x
                        best_lml = -result.fun
                except:
                    continue

            self.kernel_ = self.kernel.clone_with_theta(best_theta)
            self.log_marginal_likelihood_value_ = best_lml

        def _log_marginal_likelihood(self) -> float:
            """Compute log marginal likelihood."""
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha

            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return -1e25

            # Compute log marginal likelihood
            alpha = solve_triangular(L, self.y_train_, lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)

            # Data fit term
            data_fit = 0.5 * alpha.T @ self.y_train_

            # Complexity penalty
            complexity_penalty = np.sum(np.log(np.diag(L)))

            # Normalization constant
            normalization = 0.5 * len(self.y_train_) * np.log(2 * np.pi)

            return -(data_fit + complexity_penalty + normalization)

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            """
            Compute the negative log-likelihood score.

            Args:
                X: Test features
                y: Test targets

            Returns:
                Negative log-likelihood score
            """
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before scoring")

            X, y = np.asarray(X), np.asarray(y)

            # Get predictions
            mean, std = self.predict(X, return_std=True)

            # Compute negative log-likelihood
            nll = 0.5 * np.sum((y - mean)**2 / std**2 + np.log(2 * np.pi * std**2))

            return nll
    ```

=== "Sparse GP Regression"
    ```python
    class SparseGPRegressor:
        """
        Sparse Gaussian Process Regression using inducing points.
        """

        def __init__(self, n_inducing=100, kernel=None, alpha=1e-6):
            self.n_inducing = n_inducing
            self.kernel = kernel
            self.alpha = alpha
            self.X_inducing_ = None
            self.X_train_ = None
            self.y_train_ = None
            self.kernel_ = None
            self.L_ = None
            self.alpha_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseGPRegressor':
            """Fit sparse GP regressor."""
            X, y = np.asarray(X), np.asarray(y)

            self.X_train_ = X
            self.y_train_ = y

            # Select inducing points (simplified - could use k-means or other methods)
            n_samples = len(X)
            if n_samples <= self.n_inducing:
                self.X_inducing_ = X.copy()
            else:
                indices = np.random.choice(n_samples, self.n_inducing, replace=False)
                self.X_inducing_ = X[indices]

            # Initialize kernel
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)

            self.kernel_ = self.kernel

            # Optimize hyperparameters (simplified)
            self._optimize_hyperparameters()

            # Compute sparse approximation
            self._fit_sparse_model()

            return self

        def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Predict using sparse GP approximation."""
            if self.X_inducing_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)

            # Compute sparse kernel matrices
            K_mm = self.kernel_(self.X_inducing_)  # inducing-inducing
            K_nm = self.kernel_(self.X_train_, self.X_inducing_)  # train-inducing
            K_star = self.kernel_(self.X_inducing_, X)  # inducing-test

            # Add noise to diagonal
            K_mm[np.diag_indices_from(K_mm)] += self.alpha

            # Compute sparse approximation
            L_mm = cholesky(K_mm, lower=True)
            A = solve_triangular(L_mm, K_nm.T, lower=True)
            B = A @ A.T + np.eye(self.n_inducing)
            L_B = cholesky(B, lower=True)

            # Compute mean and variance
            v = solve_triangular(L_mm, K_star, lower=True)
            mean = v.T @ solve_triangular(L_B, A @ self.y_train_, lower=True)

            if return_std:
                v2 = solve_triangular(L_B, v, lower=True)
                var = np.diag(self.kernel_(X)) - np.sum(v2**2, axis=0)
                var = np.maximum(var, 1e-10)
                std = np.sqrt(var)
                return mean, std
            else:
                return mean, None

        def _fit_sparse_model(self):
            """Fit the sparse GP model."""
            # Compute sparse kernel matrices
            K_mm = self.kernel_(self.X_inducing_)
            K_nm = self.kernel_(self.X_train_, self.X_inducing_)

            # Add noise to diagonal
            K_mm[np.diag_indices_from(K_mm)] += self.alpha

            # Compute sparse approximation
            L_mm = cholesky(K_mm, lower=True)
            A = solve_triangular(L_mm, K_nm.T, lower=True)
            B = A @ A.T + np.eye(self.n_inducing)
            L_B = cholesky(B, lower=True)

            # Store for predictions
            self.L_ = L_mm
            self.alpha_ = solve_triangular(L_B, A @ self.y_train_, lower=True)

        def _optimize_hyperparameters(self):
            """Optimize hyperparameters (simplified implementation)."""
            # This would implement proper hyperparameter optimization
            # for the sparse GP case
            pass
    ```

=== "Multi-output GP Regression"
    ```python
    class MultiOutputGPRegressor:
        """
        Multi-output Gaussian Process Regression.
        """

        def __init__(self, kernel=None, alpha=1e-6, optimizer='L-BFGS-B', n_restarts=10):
            self.kernel = kernel
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts = n_restarts
            self.X_train_ = None
            self.y_train_ = None
            self.kernels_ = []
            self.Ls_ = []
            self.alphas_ = []
            self.n_outputs_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputGPRegressor':
            """Fit multi-output GP regressor."""
            X, y = np.asarray(X), np.asarray(y)

            if y.ndim == 1:
                y = y.reshape(-1, 1)

            self.X_train_ = X
            self.y_train_ = y
            self.n_outputs_ = y.shape[1]

            # Initialize kernels for each output
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)

            self.kernels_ = [self.kernel.clone() for _ in range(self.n_outputs_)]

            # Fit separate GP for each output
            for i in range(self.n_outputs_):
                self._fit_output_gp(i)

            return self

        def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Predict using multi-output GP."""
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)
            n_samples = X.shape[0]

            # Predict for each output
            means = np.zeros((n_samples, self.n_outputs_))
            stds = np.zeros((n_samples, self.n_outputs_)) if return_std else None

            for i in range(self.n_outputs_):
                mean, std = self._predict_output(X, i, return_std)
                means[:, i] = mean
                if return_std:
                    stds[:, i] = std

            if return_std:
                return means, stds
            else:
                return means, None

        def _fit_output_gp(self, output_idx: int):
            """Fit GP for a specific output."""
            # Compute kernel matrix
            K = self.kernels_[output_idx](self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha

            # Cholesky decomposition
            L = cholesky(K, lower=True)
            self.Ls_.append(L)

            # Solve for alpha
            alpha = solve_triangular(L, self.y_train_[:, output_idx], lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)
            self.alphas_.append(alpha)

        def _predict_output(self, X: np.ndarray, output_idx: int, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Predict for a specific output."""
            # Compute kernel matrices
            K_star = self.kernels_[output_idx](self.X_train_, X)
            K_star_star = self.kernels_[output_idx](X)

            # Mean prediction
            mean = K_star.T @ self.alphas_[output_idx]

            if return_std:
                # Variance prediction
                v = solve_triangular(self.Ls_[output_idx], K_star, lower=True)
                var = K_star_star - v.T @ v
                var = np.maximum(var, 1e-10)
                std = np.sqrt(np.diag(var))
                return mean, std
            else:
                return mean, None
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/gaussian_process/gp_regression.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/gaussian_process/gp_regression.py)
    - **Tests**: [`tests/unit/gaussian_process/test_gp_regression.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/gaussian_process/test_gp_regression.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard GPR** | $O(n^3)$ | $O(n^2)$ | Full GP with n training points |
    **Sparse GPR** | $O(m^3 + nm^2)$ | $O(m^2 + nm)$ | m inducing points |
    **Multi-output GPR** | $O(d \cdot n^3)$ | $O(d \cdot n^2)$ | d outputs |

!!! warning "Performance Considerations"
    - **Cubic complexity** in training set size limits scalability
    - **Hyperparameter optimization** can be computationally expensive
    - **Memory usage** grows quadratically with training set size
    - **Numerical stability** requires careful implementation

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Time Series Forecasting"
        - **Financial Markets**: Stock price and volatility prediction
        - **Weather Forecasting**: Temperature and precipitation prediction
        - **Energy Systems**: Power demand and renewable energy prediction
        - **Economics**: GDP and inflation forecasting

    !!! grid-item "Spatial Interpolation"
        - **Geostatistics**: Soil property mapping and mineral exploration
        - **Environmental Monitoring**: Air quality and pollution mapping
        - **Meteorology**: Weather station data interpolation
        - **Epidemiology**: Disease spread modeling

    !!! grid-item "Engineering Applications"
        - **Robotics**: Sensor fusion and state estimation
        - **Control Systems**: System identification and modeling
        - **Materials Science**: Property prediction and optimization
        - **Aerospace**: Aerodynamic coefficient prediction

    !!! grid-item "Educational Value"
        - **Bayesian Methods**: Understanding probabilistic regression
        - **Non-parametric Models**: Learning flexible model learning
        - **Uncertainty Quantification**: Understanding prediction confidence
        - **Kernel Methods**: Learning kernel-based machine learning

!!! success "Educational Value"
    - **Probabilistic Regression**: Perfect example of Bayesian regression
    - **Uncertainty Quantification**: Shows how to estimate prediction confidence
    - **Non-parametric Methods**: Demonstrates flexible model learning
    - **Kernel Methods**: Illustrates kernel-based machine learning

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
        2. **Quiñonero-Candela, J., & Rasmussen, C. E.** (2005). A unifying view of sparse approximate Gaussian process regression. *Journal of Machine Learning Research*, 6, 1939-1959.

    !!! grid-item "Gaussian Process Textbooks"
        3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
        4. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

    !!! grid-item "Online Resources"
        5. [Gaussian Process Regression - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_process)
        6. [GPyTorch Documentation](https://docs.gpytorch.ai/) - PyTorch-based GP library
        7. [scikit-learn GP Documentation](https://scikit-learn.org/stable/modules/gaussian_process.html)

    !!! grid-item "Implementation & Practice"
        8. [GPy Library](https://github.com/SheffieldML/GPy) - Python GP library
        9. [GPflow Library](https://github.com/GPflow/GPflow) - TensorFlow-based GP library
        10. [GPyTorch Library](https://github.com/cornellius-gp/gpytorch) - PyTorch-based GP library

!!! tip "Interactive Learning"
    Try implementing GP Regression yourself! Start with simple 1D and 2D datasets to understand how the algorithm works. Experiment with different kernel functions to see how they affect the model behavior. Try implementing sparse GP regression to understand how to handle larger datasets. Compare GP regression with other regression methods to see the benefits of uncertainty quantification. This will give you deep insight into probabilistic regression and kernel-based machine learning.

## Navigation

{{ nav_grid(current_algorithm="gp-regression", current_family="gaussian-process", max_related=5) }}
