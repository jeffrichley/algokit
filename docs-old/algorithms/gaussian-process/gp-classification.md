---
algorithm_key: "gp-classification"
tags: [gaussian-process, algorithms, gp-classification, bayesian-inference, probabilistic-classification]
title: "Gaussian Process Classification"
family: "gaussian-process"
---

# Gaussian Process Classification

{{ algorithm_card("gp-classification") }}

!!! abstract "Overview"
    Gaussian Process Classification (GPC) is a powerful probabilistic approach to classification that extends Gaussian Process regression to handle discrete class labels. Unlike traditional classification methods, GPC provides not only class predictions but also uncertainty estimates, making it particularly valuable for applications where confidence in predictions is important.

    The algorithm works by modeling the latent function that maps inputs to class probabilities using a Gaussian Process, then applying a sigmoid or probit function to convert the latent values to valid probabilities. This approach naturally handles uncertainty and provides principled Bayesian inference for classification problems.

## Mathematical Formulation

!!! math "Gaussian Process Classification"
    In GPC, we model the latent function $f(x)$ as a Gaussian Process:

    $$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

    The class probabilities are obtained through a link function:

    $$p(y = 1 | x) = \sigma(f(x))$$

    Where $\sigma$ is typically the sigmoid function:

    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    For binary classification, the likelihood is:

    $$p(y | f) = \prod_{i=1}^n \sigma(f(x_i))^{y_i} (1 - \sigma(f(x_i)))^{1-y_i}$$

    The posterior is approximated using methods like Laplace approximation or Expectation Propagation.

!!! success "Key Properties"
    - **Probabilistic**: Provides uncertainty estimates for predictions
    - **Non-parametric**: No fixed model structure assumptions
    - **Bayesian**: Incorporates prior knowledge through kernel choice
    - **Flexible**: Can handle complex decision boundaries

## Implementation Approaches

=== "Basic GP Classification (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import cholesky, solve_triangular
    from scipy.special import expit
    from typing import Tuple, Optional
    import warnings

    class GaussianProcessClassifier:
        """
        Gaussian Process Classification implementation.

        Args:
            kernel: Kernel function for the GP
            alpha: Regularization parameter
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

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessClassifier':
            """
            Fit the Gaussian Process classifier.

            Args:
                X: Training features of shape (n_samples, n_features)
                y: Training labels of shape (n_samples,)

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

            # Check for binary classification
            unique_classes = np.unique(y)
            if len(unique_classes) != 2:
                raise ValueError("Only binary classification is supported")

            # Convert labels to {0, 1}
            self.classes_ = unique_classes
            y_binary = (y == unique_classes[1]).astype(int)

            self.X_train_ = X
            self.y_train_ = y_binary

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

            # Compute alpha using Newton's method
            self.alpha_ = self._newton_method()

            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """
            Predict class probabilities.

            Args:
                X: Test features of shape (n_samples, n_features)

            Returns:
                Class probabilities of shape (n_samples, 2)
            """
            if self.X_train_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)

            # Compute mean and variance of latent function
            K_star = self.kernel_(self.X_train_, X)
            K_star_star = self.kernel_(X)

            # Solve for mean
            v = solve_triangular(self.L_, K_star, lower=True)
            f_mean = K_star.T @ self.alpha_

            # Solve for variance
            v2 = solve_triangular(self.L_, v, lower=True)
            f_var = K_star_star - v.T @ v

            # Ensure variance is positive
            f_var = np.maximum(f_var, 1e-10)

            # Convert to probabilities using sigmoid
            f_std = np.sqrt(f_var)
            prob_positive = self._sigmoid_prob(f_mean, f_std)
            prob_negative = 1 - prob_positive

            return np.column_stack([prob_negative, prob_positive])

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predict class labels.

            Args:
                X: Test features of shape (n_samples, n_features)

            Returns:
                Predicted class labels
            """
            proba = self.predict_proba(X)
            predictions = self.classes_[(proba[:, 1] > 0.5).astype(int)]
            return predictions

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
            alpha = self._newton_method()

            # Data fit term
            data_fit = 0.5 * alpha.T @ K @ alpha

            # Complexity penalty
            complexity_penalty = np.sum(np.log(np.diag(L)))

            # Normalization constant
            normalization = 0.5 * len(self.X_train_) * np.log(2 * np.pi)

            return -(data_fit + complexity_penalty + normalization)

        def _newton_method(self, max_iter: int = 20) -> np.ndarray:
            """Newton's method for finding mode of posterior."""
            n = len(self.X_train_)
            alpha = np.zeros(n)

            for _ in range(max_iter):
                # Compute kernel matrix
                K = self.kernel_(self.X_train_)
                K[np.diag_indices_from(K)] += self.alpha

                try:
                    L = cholesky(K, lower=True)
                except np.linalg.LinAlgError:
                    break

                # Compute predictions
                f = K @ alpha
                pi = expit(f)

                # Compute gradient and Hessian
                W = pi * (1 - pi)
                W_sqrt = np.sqrt(W)

                # Gradient
                grad = self.y_train_ - pi

                # Hessian
                B = np.eye(n) + W_sqrt[:, None] * K * W_sqrt[None, :]
                L_B = cholesky(B, lower=True)

                # Solve for update
                b = W_sqrt * (K @ grad + f)
                a = b - solve_triangular(L_B, W_sqrt[:, None] * K @ solve_triangular(L_B, b, lower=True), lower=True)
                alpha = K @ a

                # Check convergence
                if np.linalg.norm(grad) < 1e-6:
                    break

            return alpha

        def _sigmoid_prob(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            """Compute sigmoid probability with uncertainty."""
            # Use approximation for sigmoid with uncertainty
            # This is a simplified version - more sophisticated methods exist
            return expit(mean / np.sqrt(1 + np.pi * std**2 / 8))
    ```

=== "Multi-class GP Classification"
    ```python
    class MultiClassGPClassifier:
        """
        Multi-class Gaussian Process Classification using one-vs-rest.
        """

        def __init__(self, kernel=None, alpha=1e-6, optimizer='L-BFGS-B', n_restarts=10):
            self.kernel = kernel
            self.alpha = alpha
            self.optimizer = optimizer
            self.n_restarts = n_restarts
            self.classifiers_ = []
            self.classes_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassGPClassifier':
            """Fit multi-class GP classifier using one-vs-rest."""
            X, y = np.asarray(X), np.asarray(y)

            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.classifiers_ = []

            # Train one binary classifier per class
            for i, class_label in enumerate(self.classes_):
                # Create binary labels for this class
                y_binary = (y == class_label).astype(int)

                # Train binary classifier
                classifier = GaussianProcessClassifier(
                    kernel=self.kernel,
                    alpha=self.alpha,
                    optimizer=self.optimizer,
                    n_restarts=self.n_restarts
                )
                classifier.fit(X, y_binary)
                self.classifiers_.append(classifier)

            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities for multi-class problem."""
            if self.classifiers_ is None:
                raise ValueError("Model must be fitted before making predictions")

            X = np.asarray(X)
            n_samples = X.shape[0]
            n_classes = len(self.classes_)

            # Get probabilities from each binary classifier
            proba = np.zeros((n_samples, n_classes))

            for i, classifier in enumerate(self.classifiers_):
                proba[:, i] = classifier.predict_proba(X)[:, 1]

            # Normalize probabilities
            proba_sum = proba.sum(axis=1, keepdims=True)
            proba_sum[proba_sum == 0] = 1  # Avoid division by zero
            proba = proba / proba_sum

            return proba

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels for multi-class problem."""
            proba = self.predict_proba(X)
            predictions = self.classes_[np.argmax(proba, axis=1)]
            return predictions
    ```

=== "Sparse GP Classification"
    ```python
    class SparseGPClassifier:
        """
        Sparse Gaussian Process Classification using inducing points.
        """

        def __init__(self, n_inducing=100, kernel=None, alpha=1e-6):
            self.n_inducing = n_inducing
            self.kernel = kernel
            self.alpha = alpha
            self.X_inducing_ = None
            self.X_train_ = None
            self.y_train_ = None
            self.kernel_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseGPClassifier':
            """Fit sparse GP classifier."""
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

            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities using sparse approximation."""
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

            v2 = solve_triangular(L_B, v, lower=True)
            var = np.diag(self.kernel_(X)) - np.sum(v2**2, axis=0)
            var = np.maximum(var, 1e-10)

            # Convert to probabilities
            std = np.sqrt(var)
            prob_positive = self._sigmoid_prob(mean, std)
            prob_negative = 1 - prob_positive

            return np.column_stack([prob_negative, prob_positive])

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            proba = self.predict_proba(X)
            return (proba[:, 1] > 0.5).astype(int)

        def _optimize_hyperparameters(self):
            """Optimize hyperparameters (simplified implementation)."""
            # This would implement proper hyperparameter optimization
            # for the sparse GP case
            pass

        def _sigmoid_prob(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            """Compute sigmoid probability with uncertainty."""
            return expit(mean / np.sqrt(1 + np.pi * std**2 / 8))
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/gaussian_process/gp_classification.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/gaussian_process/gp_classification.py)
    - **Tests**: [`tests/unit/gaussian_process/test_gp_classification.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/gaussian_process/test_gp_classification.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard GPC** | $O(n^3)$ | $O(n^2)$ | Full GP with n training points |
    **Multi-class GPC** | $O(C \cdot n^3)$ | $O(C \cdot n^2)$ | C classes, one-vs-rest |
    **Sparse GPC** | $O(m^3 + nm^2)$ | $O(m^2 + nm)$ | m inducing points |

!!! warning "Performance Considerations"
    - **Cubic complexity** in training set size limits scalability
    - **Hyperparameter optimization** can be computationally expensive
    - **Memory usage** grows quadratically with training set size
    - **Numerical stability** requires careful implementation

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Medical Diagnosis"
        - **Disease Classification**: Probabilistic disease prediction
        - **Drug Discovery**: Compound classification and screening
        - **Medical Imaging**: Image classification with uncertainty
        - **Clinical Decision Support**: Risk assessment and diagnosis

    !!! grid-item "Computer Vision"
        - **Image Classification**: Object recognition with confidence
        - **Face Recognition**: Identity verification with uncertainty
        - **Medical Imaging**: Automated diagnosis and screening
        - **Quality Control**: Defect detection in manufacturing

    !!! grid-item "Natural Language Processing"
        - **Sentiment Analysis**: Text classification with confidence
        - **Spam Detection**: Email filtering with uncertainty
        - **Topic Classification**: Document categorization
        - **Language Detection**: Text language identification

    !!! grid-item "Educational Value"
        - **Bayesian Methods**: Understanding probabilistic classification
        - **Gaussian Processes**: Learning non-parametric models
        - **Uncertainty Quantification**: Understanding prediction confidence
        - **Kernel Methods**: Learning kernel-based approaches

!!! success "Educational Value"
    - **Probabilistic Classification**: Perfect example of Bayesian classification
    - **Uncertainty Quantification**: Shows how to estimate prediction confidence
    - **Non-parametric Methods**: Demonstrates flexible model learning
    - **Kernel Methods**: Illustrates kernel-based machine learning

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
        2. **Nickisch, H., & Rasmussen, C. E.** (2008). Approximations for binary Gaussian process classification. *Journal of Machine Learning Research*, 9, 2035-2078.

    !!! grid-item "Gaussian Process Textbooks"
        3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
        4. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

    !!! grid-item "Online Resources"
        5. [Gaussian Process Classification - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_process_classification)
        6. [GPyTorch Documentation](https://docs.gpytorch.ai/) - PyTorch-based GP library
        7. [scikit-learn GP Documentation](https://scikit-learn.org/stable/modules/gaussian_process.html)

    !!! grid-item "Implementation & Practice"
        8. [GPy Library](https://github.com/SheffieldML/GPy) - Python GP library
        9. [GPflow Library](https://github.com/GPflow/GPflow) - TensorFlow-based GP library
        10. [GPyTorch Library](https://github.com/cornellius-gp/gpytorch) - PyTorch-based GP library

!!! tip "Interactive Learning"
    Try implementing GP Classification yourself! Start with simple 2D datasets to understand how the algorithm works. Experiment with different kernel functions to see how they affect the decision boundaries. Try implementing sparse GP classification to understand how to handle larger datasets. Compare GP classification with other probabilistic methods to see the benefits of the Gaussian Process approach. This will give you deep insight into probabilistic classification and uncertainty quantification.

## Navigation

{{ nav_grid(current_algorithm="gp-classification", current_family="gaussian-process", max_related=5) }}
