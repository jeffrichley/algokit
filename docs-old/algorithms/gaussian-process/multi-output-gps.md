---
algorithm_key: "multi-output-gps"
tags: [gaussian-process, algorithms, multi-output-gps, multi-task-learning, correlated-outputs, transfer-learning]
title: "Multi-output Gaussian Processes"
family: "gaussian-process"
---

# Multi-output Gaussian Processes

{{ algorithm_card("multi-output-gps") }}

!!! abstract "Overview"
    Multi-output Gaussian Processes (MOGPs) extend standard GPs to handle multiple correlated outputs simultaneously. Unlike treating each output independently, MOGPs model the correlations between outputs, enabling knowledge transfer and improved predictions, especially when some outputs have limited data.

    The algorithm works by defining a multi-output kernel that captures both input similarities and output correlations. This allows the model to leverage information from all outputs to make better predictions, making MOGPs particularly valuable for multi-task learning, transfer learning, and scenarios where outputs are naturally correlated.

## Mathematical Formulation

!!! math "Multi-output Gaussian Process"
    A Multi-output GP with $D$ outputs is defined as:

    $$f_d(x) \sim \mathcal{GP}(0, k_d(x, x'))$$

    Where $f_d(x)$ is the $d$-th output function and $k_d(x, x')$ is the kernel for output $d$.

    The key insight is to model correlations between outputs using a cross-covariance function:

    $$\text{Cov}[f_d(x), f_{d'}(x')] = k_{dd'}(x, x')$$

    This can be factorized as:

    $$k_{dd'}(x, x') = \sum_{q=1}^Q A_{dq} A_{d'q} k_q(x, x')$$

    Where $A$ is a $D \times Q$ matrix and $k_q$ are $Q$ latent kernels. This is known as the Linear Model of Coregionalization (LMC).

!!! success "Key Properties"
    - **Multi-task Learning**: Leverages correlations between tasks
    - **Transfer Learning**: Knowledge transfer between related outputs
    - **Correlation Modeling**: Captures output dependencies
    - **Flexible**: Works with any number of outputs

## Implementation Approaches

=== "Basic Multi-output GP Implementation (Recommended)"
    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from scipy.linalg import cholesky, solve_triangular
    from typing import Tuple, Optional, List
    import warnings

    class MultiOutputGaussianProcess:
        """
        Multi-output Gaussian Process implementation using Linear Model of Coregionalization.

        Args:
            input_dim: Input dimension
            output_dim: Number of outputs
            n_latent: Number of latent functions
            kernel: Base kernel function
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs
        """

        def __init__(self, input_dim: int, output_dim: int, n_latent: int = None,
                     kernel=None, learning_rate=0.01, n_epochs=1000):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_latent = n_latent or output_dim
            self.kernel = kernel
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.X_train_ = None
            self.y_train_ = None
            self.kernels_ = []
            self.coregionalization_matrix = None
            self.optimizer = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputGaussianProcess':
            """
            Fit the Multi-output Gaussian Process model.

            Args:
                X: Training features of shape (n_samples, input_dim)
                y: Training targets of shape (n_samples, output_dim)

            Returns:
                self
            """
            X, y = np.asarray(X), np.asarray(y)

            # Validate inputs
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            if y.ndim != 2:
                raise ValueError("y must be 2-dimensional")
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")

            self.X_train_ = X
            self.y_train_ = y

            # Initialize components
            self._initialize_kernels()
            self._initialize_coregionalization_matrix()

            # Convert to PyTorch tensors
            X_torch = torch.FloatTensor(X)
            y_torch = torch.FloatTensor(y)

            # Initialize optimizer
            self.optimizer = optim.Adam(self._get_parameters(), lr=self.learning_rate)

            # Training loop
            for epoch in range(self.n_epochs):
                self.optimizer.zero_grad()

                # Compute loss
                loss = self._compute_loss(X_torch, y_torch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            return self

        def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """
            Predict using the Multi-output Gaussian Process.

            Args:
                X: Test features of shape (n_samples, input_dim)
                return_std: Whether to return standard deviations

            Returns:
                Tuple of (mean, std) predictions
            """
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)
            X_torch = torch.FloatTensor(X)

            with torch.no_grad():
                # Compute multi-output kernel matrix
                K_star = self._compute_multi_output_kernel(X_torch, torch.FloatTensor(self.X_train_))
                K_star_star = self._compute_multi_output_kernel(X_torch, X_torch)

                # Compute training kernel matrix
                K_train = self._compute_multi_output_kernel(torch.FloatTensor(self.X_train_), torch.FloatTensor(self.X_train_))

                # Add noise to diagonal
                K_train = K_train + 1e-6 * torch.eye(K_train.shape[0])

                # Compute Cholesky decomposition
                L = torch.cholesky(K_train)

                # Solve for mean
                alpha = torch.solve(y_torch, L)[0]
                alpha = torch.solve(alpha, L.T)[0]
                mean = K_star.T @ alpha

                if return_std:
                    # Compute variance
                    v = torch.solve(K_star, L)[0]
                    var = K_star_star - v.T @ v
                    var = torch.clamp(var, min=1e-10)
                    std = torch.sqrt(torch.diag(var))
                    return mean.numpy(), std.numpy()
                else:
                    return mean.numpy(), None

        def _initialize_kernels(self):
            """Initialize kernels for each latent function."""
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)

            self.kernels_ = [self.kernel.clone() for _ in range(self.n_latent)]

        def _initialize_coregionalization_matrix(self):
            """Initialize coregionalization matrix."""
            # Initialize A matrix (output_dim x n_latent)
            self.coregionalization_matrix = torch.randn(self.output_dim, self.n_latent, requires_grad=True)

        def _get_parameters(self):
            """Get all trainable parameters."""
            params = []
            params.append(self.coregionalization_matrix)

            # Add kernel parameters if they exist
            for kernel in self.kernels_:
                if hasattr(kernel, 'theta'):
                    params.extend(kernel.theta)

            return params

        def _compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute the negative log-likelihood."""
            # Compute multi-output kernel matrix
            K = self._compute_multi_output_kernel(X, X)

            # Add noise to diagonal
            K = K + 1e-6 * torch.eye(K.shape[0])

            # Compute Cholesky decomposition
            try:
                L = torch.cholesky(K)
            except:
                return torch.tensor(1e25)

            # Compute log-likelihood
            alpha = torch.solve(y, L)[0]
            alpha = torch.solve(alpha, L.T)[0]

            # Data fit term
            data_fit = 0.5 * torch.sum(alpha * y)

            # Complexity penalty
            complexity_penalty = torch.sum(torch.log(torch.diag(L)))

            # Normalization constant
            normalization = 0.5 * y.numel() * torch.log(2 * torch.pi)

            return -(data_fit + complexity_penalty + normalization)

        def _compute_multi_output_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
            """Compute multi-output kernel matrix."""
            n1 = X1.shape[0]
            n2 = X2.shape[0]

            # Initialize kernel matrix
            K = torch.zeros(n1 * self.output_dim, n2 * self.output_dim)

            # Compute kernel for each pair of outputs
            for d1 in range(self.output_dim):
                for d2 in range(self.output_dim):
                    # Compute cross-covariance
                    cross_cov = self._compute_cross_covariance(X1, X2, d1, d2)

                    # Fill in the appropriate block
                    start1 = d1 * n1
                    end1 = (d1 + 1) * n1
                    start2 = d2 * n2
                    end2 = (d2 + 1) * n2

                    K[start1:end1, start2:end2] = cross_cov

            return K

        def _compute_cross_covariance(self, X1: torch.Tensor, X2: torch.Tensor,
                                    output1: int, output2: int) -> torch.Tensor:
            """Compute cross-covariance between two outputs."""
            cross_cov = torch.zeros(X1.shape[0], X2.shape[0])

            # Sum over latent functions
            for q in range(self.n_latent):
                # Compute base kernel
                K_q = self._compute_base_kernel(X1, X2, q)

                # Multiply by coregionalization coefficients
                A1q = self.coregionalization_matrix[output1, q]
                A2q = self.coregionalization_matrix[output2, q]

                cross_cov += A1q * A2q * K_q

            return cross_cov

        def _compute_base_kernel(self, X1: torch.Tensor, X2: torch.Tensor, latent_idx: int) -> torch.Tensor:
            """Compute base kernel for a specific latent function."""
            # RBF kernel implementation
            X1_expanded = X1.unsqueeze(1)
            X2_expanded = X2.unsqueeze(0)

            squared_dist = torch.sum((X1_expanded - X2_expanded)**2, dim=2)
            K = torch.exp(-0.5 * squared_dist)

            return K
    ```

=== "Convolutional Multi-output GP"
    ```python
    class ConvolutionalMultiOutputGP(MultiOutputGaussianProcess):
        """
        Convolutional Multi-output Gaussian Process for image data.
        """

        def __init__(self, input_shape: Tuple[int, int, int], output_dim: int,
                     n_latent: int = None, kernel=None, learning_rate=0.01, n_epochs=1000):
            self.input_shape = input_shape
            super().__init__(np.prod(input_shape), output_dim, n_latent, kernel, learning_rate, n_epochs)

        def _compute_base_kernel(self, X1: torch.Tensor, X2: torch.Tensor, latent_idx: int) -> torch.Tensor:
            """Compute convolutional kernel for image data."""
            # Reshape to image format
            X1_images = X1.view(-1, *self.input_shape)
            X2_images = X2.view(-1, *self.input_shape)

            # Apply convolution
            conv = nn.Conv2d(self.input_shape[0], 1, kernel_size=3, padding=1)
            X1_conv = conv(X1_images)
            X2_conv = conv(X2_images)

            # Flatten for kernel computation
            X1_flat = X1_conv.view(X1_conv.shape[0], -1)
            X2_flat = X2_conv.view(X2_conv.shape[0], -1)

            # Compute RBF kernel
            X1_expanded = X1_flat.unsqueeze(1)
            X2_expanded = X2_flat.unsqueeze(0)

            squared_dist = torch.sum((X1_expanded - X2_expanded)**2, dim=2)
            K = torch.exp(-0.5 * squared_dist)

            return K
    ```

=== "Sparse Multi-output GP"
    ```python
    class SparseMultiOutputGP:
        """
        Sparse Multi-output Gaussian Process using inducing points.
        """

        def __init__(self, input_dim: int, output_dim: int, n_latent: int = None,
                     n_inducing: int = 100, kernel=None, learning_rate=0.01, n_epochs=1000):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_latent = n_latent or output_dim
            self.n_inducing = n_inducing
            self.kernel = kernel
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.X_train_ = None
            self.y_train_ = None
            self.X_inducing_ = None
            self.kernels_ = []
            self.coregionalization_matrix = None
            self.variational_params = None
            self.optimizer = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseMultiOutputGP':
            """Fit sparse multi-output GP."""
            X, y = np.asarray(X), np.asarray(y)

            if y.ndim == 1:
                y = y.reshape(-1, 1)

            self.X_train_ = X
            self.y_train_ = y

            # Initialize components
            self._initialize_kernels()
            self._initialize_coregionalization_matrix()
            self._initialize_inducing_points()
            self._initialize_variational_params()

            # Convert to PyTorch tensors
            X_torch = torch.FloatTensor(X)
            y_torch = torch.FloatTensor(y)
            X_inducing_torch = torch.FloatTensor(self.X_inducing_)

            # Initialize optimizer
            self.optimizer = optim.Adam(self._get_parameters(), lr=self.learning_rate)

            # Training loop
            for epoch in range(self.n_epochs):
                self.optimizer.zero_grad()

                # Compute loss
                loss = self._compute_loss(X_torch, y_torch, X_inducing_torch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            return self

        def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """Predict using sparse multi-output GP."""
            if self.X_inducing_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)
            X_torch = torch.FloatTensor(X)
            X_inducing_torch = torch.FloatTensor(self.X_inducing_)

            with torch.no_grad():
                # Get variational parameters
                q_mu, q_sqrt = self.variational_params

                # Compute kernel matrices
                K_mm = self._compute_multi_output_kernel(X_inducing_torch, X_inducing_torch)
                K_nm = self._compute_multi_output_kernel(X_torch, X_inducing_torch)
                K_nn = self._compute_multi_output_kernel(X_torch, X_torch)

                # Add noise to diagonal
                K_mm = K_mm + 1e-6 * torch.eye(K_mm.shape[0])

                # Compute sparse approximation
                L_mm = torch.cholesky(K_mm)
                A = torch.solve(K_nm.T, L_mm)[0]
                B = A @ A.T + torch.eye(self.n_inducing * self.output_dim)
                L_B = torch.cholesky(B)

                # Compute mean and variance
                v = torch.solve(K_nm.T, L_mm)[0]
                mean = v.T @ torch.solve(q_mu, L_B)[0]

                if return_std:
                    v2 = torch.solve(v, L_B)[0]
                    var = torch.diag(K_nn) - torch.sum(v2**2, dim=0)
                    var = torch.clamp(var, min=1e-10)
                    std = torch.sqrt(var)
                    return mean.numpy(), std.numpy()
                else:
                    return mean.numpy(), None

        def _initialize_inducing_points(self):
            """Initialize inducing points."""
            n_samples = len(self.X_train_)

            if n_samples <= self.n_inducing:
                self.X_inducing_ = self.X_train_.copy()
                self.n_inducing = n_samples
            else:
                indices = np.random.choice(n_samples, self.n_inducing, replace=False)
                self.X_inducing_ = self.X_train_[indices]

        def _initialize_variational_params(self):
            """Initialize variational parameters."""
            # Variational mean
            q_mu = torch.randn(self.n_inducing * self.output_dim, requires_grad=True)

            # Variational covariance (Cholesky factor)
            q_sqrt = torch.randn(self.n_inducing * self.output_dim,
                               self.n_inducing * self.output_dim, requires_grad=True)
            q_sqrt = torch.tril(q_sqrt)

            self.variational_params = (q_mu, q_sqrt)

        def _get_parameters(self):
            """Get all trainable parameters."""
            params = []
            params.append(self.coregionalization_matrix)
            params.append(self.variational_params[0])
            params.append(self.variational_params[1])

            return params

        def _compute_loss(self, X: torch.Tensor, y: torch.Tensor, X_inducing: torch.Tensor) -> torch.Tensor:
            """Compute variational lower bound."""
            # Get variational parameters
            q_mu, q_sqrt = self.variational_params

            # Compute kernel matrices
            K_mm = self._compute_multi_output_kernel(X_inducing, X_inducing)
            K_nm = self._compute_multi_output_kernel(X, X_inducing)
            K_nn = self._compute_multi_output_kernel(X, X)

            # Add noise to diagonal
            K_mm = K_mm + 1e-6 * torch.eye(K_mm.shape[0])

            # Compute sparse approximation
            L_mm = torch.cholesky(K_mm)
            A = torch.solve(K_nm.T, L_mm)[0]
            B = A @ A.T + torch.eye(self.n_inducing * self.output_dim)
            L_B = torch.cholesky(B)

            # Compute mean and variance
            v = torch.solve(K_nm.T, L_mm)[0]
            mean = v.T @ torch.solve(q_mu, L_B)[0]

            v2 = torch.solve(v, L_B)[0]
            var = torch.diag(K_nn) - torch.sum(v2**2, dim=0)
            var = torch.clamp(var, min=1e-10)

            # Likelihood term
            likelihood = -0.5 * torch.sum((y.view(-1) - mean)**2 / var) - 0.5 * torch.sum(torch.log(2 * torch.pi * var))

            # KL divergence term
            kl_divergence = self._compute_kl_divergence(q_mu, q_sqrt, K_mm)

            # Variational lower bound
            elbo = likelihood - kl_divergence

            return -elbo

        def _compute_kl_divergence(self, q_mu: torch.Tensor, q_sqrt: torch.Tensor, K_mm: torch.Tensor) -> torch.Tensor:
            """Compute KL divergence."""
            q_cov = q_sqrt @ q_sqrt.T

            kl = 0.5 * torch.trace(torch.solve(q_cov, K_mm)[0])
            kl += 0.5 * q_mu.T @ torch.solve(q_mu, K_mm)[0]
            kl -= 0.5 * (self.n_inducing * self.output_dim)
            kl += 0.5 * torch.logdet(K_mm) - 0.5 * torch.logdet(q_cov)

            return kl

        def _compute_multi_output_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
            """Compute multi-output kernel matrix."""
            n1 = X1.shape[0]
            n2 = X2.shape[0]

            # Initialize kernel matrix
            K = torch.zeros(n1 * self.output_dim, n2 * self.output_dim)

            # Compute kernel for each pair of outputs
            for d1 in range(self.output_dim):
                for d2 in range(self.output_dim):
                    # Compute cross-covariance
                    cross_cov = self._compute_cross_covariance(X1, X2, d1, d2)

                    # Fill in the appropriate block
                    start1 = d1 * n1
                    end1 = (d1 + 1) * n1
                    start2 = d2 * n2
                    end2 = (d2 + 1) * n2

                    K[start1:end1, start2:end2] = cross_cov

            return K

        def _compute_cross_covariance(self, X1: torch.Tensor, X2: torch.Tensor,
                                    output1: int, output2: int) -> torch.Tensor:
            """Compute cross-covariance between two outputs."""
            cross_cov = torch.zeros(X1.shape[0], X2.shape[0])

            # Sum over latent functions
            for q in range(self.n_latent):
                # Compute base kernel
                K_q = self._compute_base_kernel(X1, X2, q)

                # Multiply by coregionalization coefficients
                A1q = self.coregionalization_matrix[output1, q]
                A2q = self.coregionalization_matrix[output2, q]

                cross_cov += A1q * A2q * K_q

            return cross_cov

        def _compute_base_kernel(self, X1: torch.Tensor, X2: torch.Tensor, latent_idx: int) -> torch.Tensor:
            """Compute base kernel for a specific latent function."""
            X1_expanded = X1.unsqueeze(1)
            X2_expanded = X2.unsqueeze(0)

            squared_dist = torch.sum((X1_expanded - X2_expanded)**2, dim=2)
            K = torch.exp(-0.5 * squared_dist)

            return K
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/gaussian_process/multi_output_gps.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/gaussian_process/multi_output_gps.py)
    - **Tests**: [`tests/unit/gaussian_process/test_multi_output_gps.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/gaussian_process/test_multi_output_gps.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard MOGP** | $O(D^2 \cdot n^3)$ | $O(D^2 \cdot n^2)$ | D outputs, n training points |
    **Convolutional MOGP** | $O(D^2 \cdot n^3 + D^2 \cdot n \cdot d^2)$ | $O(D^2 \cdot n^2)$ | d = image dimension |
    **Sparse MOGP** | $O(D^2 \cdot m^3 + D^2 \cdot nm^2)$ | $O(D^2 \cdot m^2 + D^2 \cdot nm)$ | m inducing points |

!!! warning "Performance Considerations"
    - **Quadratic scaling** in number of outputs limits scalability
    - **Coregionalization matrix** requires careful initialization
    - **Memory usage** grows quadratically with outputs
    - **Training stability** can be challenging

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Multi-task Learning"
        - **Computer Vision**: Object detection and classification
        - **Natural Language Processing**: Sentiment and topic analysis
        - **Robotics**: Multi-objective control and planning
        - **Healthcare**: Multi-symptom diagnosis and treatment

    !!! grid-item "Transfer Learning"
        - **Domain Adaptation**: Cross-domain knowledge transfer
        - **Few-shot Learning**: Learning with limited data
        - **Continual Learning**: Sequential task learning
        - **Meta-learning**: Learning to learn

    !!! grid-item "Correlated Outputs"
        - **Time Series**: Multiple related time series
        - **Spatial Data**: Multiple spatial variables
        - **Sensor Networks**: Multiple sensor readings
        - **Financial Modeling**: Multiple asset prices

    !!! grid-item "Educational Value"
        - **Multi-task Learning**: Understanding task relationships
        - **Transfer Learning**: Learning knowledge transfer
        - **Correlation Modeling**: Understanding output dependencies
        - **Model Composition**: Learning complex model structures

!!! success "Educational Value"
    - **Multi-task Learning**: Perfect example of learning multiple related tasks
    - **Transfer Learning**: Shows how to leverage knowledge across tasks
    - **Correlation Modeling**: Demonstrates how to model output dependencies
    - **Model Composition**: Illustrates how to build complex models

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Alvarez, M. A., & Lawrence, N. D.** (2011). Computationally efficient convolved multiple output Gaussian processes. *Journal of Machine Learning Research*, 12, 1459-1500.
        2. **Bonilla, E. V., Chai, K. M., & Williams, C. K.** (2008). Multi-task Gaussian process prediction. *NIPS*, 20, 153-160.

    !!! grid-item "Multi-task Learning Textbooks"
        3. **Caruana, R.** (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
        4. **Zhang, Y., & Yang, Q.** (2017). A survey on multi-task learning. *arXiv preprint arXiv:1707.08114*.

    !!! grid-item "Online Resources"
        5. [Multi-output Gaussian Processes - Wikipedia](https://en.wikipedia.org/wiki/Multi-output_Gaussian_process)
        6. [GPyTorch Multi-output GP Tutorial](https://docs.gpytorch.ai/en/latest/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html)
        7. [Multi-task Learning Tutorial](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/10701-s07/www/lectures/lecture_12.pdf)

    !!! grid-item "Implementation & Practice"
        8. [GPyTorch Library](https://github.com/cornellius-gp/gpytorch) - PyTorch-based GP library
        9. [GPflow Library](https://github.com/GPflow/GPflow) - TensorFlow-based GP library
        10. [GPy Library](https://github.com/SheffieldML/GPy) - Python GP library

!!! tip "Interactive Learning"
    Try implementing Multi-output GPs yourself! Start with simple 2-output problems to understand how the algorithm works. Experiment with different coregionalization matrices to see how they affect the model behavior. Try implementing sparse multi-output GPs to understand how to handle larger datasets. Compare multi-output GPs with independent GPs to see the benefits of modeling correlations. This will give you deep insight into multi-task learning and correlation modeling.

## Navigation

{{ nav_grid(current_algorithm="multi-output-gps", current_family="gaussian-process", max_related=5) }}
