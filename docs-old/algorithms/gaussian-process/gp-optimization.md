---
algorithm_key: "gp-optimization"
tags: [gaussian-process, algorithms, gp-optimization, bayesian-optimization, acquisition-functions, global-optimization]
title: "Gaussian Process Optimization"
family: "gaussian-process"
---

# Gaussian Process Optimization

{{ algorithm_card("gp-optimization") }}

!!! abstract "Overview"
    Gaussian Process Optimization (GPO) is a powerful approach to global optimization that uses Gaussian Processes to model expensive-to-evaluate objective functions. Unlike traditional optimization methods that require gradients or many function evaluations, GPO is particularly effective for black-box optimization problems where function evaluations are costly.

    The algorithm works by building a probabilistic model of the objective function using previously evaluated points, then using acquisition functions to intelligently select the next point to evaluate. This approach balances exploration (searching in unexplored regions) with exploitation (focusing on promising areas), making it highly efficient for optimization problems with limited budgets.

## Mathematical Formulation

!!! math "Gaussian Process Optimization"
    Given an objective function $f(x)$ to minimize, GPO models it as a Gaussian Process:
    
    $$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$
    
    The acquisition function $\alpha(x)$ guides the search:
    
    $$\alpha(x) = \mathbb{E}[f(x)] - \beta \cdot \text{std}(f(x))$$
    
    Common acquisition functions include:
    
    - **Expected Improvement (EI)**: $\alpha_{EI}(x) = \mathbb{E}[\max(0, f_{min} - f(x))]$
    - **Upper Confidence Bound (UCB)**: $\alpha_{UCB}(x) = \mu(x) + \beta \sigma(x)$
    - **Probability of Improvement (PI)**: $\alpha_{PI}(x) = P(f(x) < f_{min})$
    
    Where $f_{min}$ is the current best function value.

!!! success "Key Properties"
    - **Global Optimization**: Finds global optima, not just local ones
    - **Sample Efficient**: Requires fewer function evaluations than grid search
    - **Uncertainty Aware**: Incorporates uncertainty in decision making
    - **Flexible**: Works with any acquisition function

## Implementation Approaches

=== "Basic GP Optimization (Recommended)"
    ```python
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import cholesky, solve_triangular
    from typing import Callable, Tuple, Optional, List
    import warnings
    
    class GaussianProcessOptimizer:
        """
        Gaussian Process Optimization implementation.
        
        Args:
            kernel: Kernel function for the GP
            acquisition_func: Acquisition function to use
            beta: Exploration parameter for UCB
            n_restarts: Number of optimization restarts
            bounds: Bounds for the optimization domain
        """
        
        def __init__(self, kernel=None, acquisition_func='EI', beta=2.0, 
                     n_restarts=10, bounds=None):
            self.kernel = kernel
            self.acquisition_func = acquisition_func
            self.beta = beta
            self.n_restarts = n_restarts
            self.bounds = bounds
            self.X_train_ = None
            self.y_train_ = None
            self.kernel_ = None
            self.L_ = None
            self.alpha_ = None
            self.f_min_ = None
        
        def optimize(self, objective_func: Callable, n_iterations: int, 
                    X_init: Optional[np.ndarray] = None, 
                    y_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
            """
            Optimize the objective function using GP optimization.
            
            Args:
                objective_func: Function to optimize
                n_iterations: Number of optimization iterations
                X_init: Initial training points
                y_init: Initial function values
                
            Returns:
                Tuple of (best_point, best_value)
            """
            # Initialize with random points if not provided
            if X_init is None:
                X_init = self._random_points(5)
                y_init = np.array([objective_func(x) for x in X_init])
            
            self.X_train_ = X_init.copy()
            self.y_train_ = y_init.copy()
            self.f_min_ = np.min(y_init)
            
            # Optimize hyperparameters
            self._optimize_hyperparameters()
            
            # Main optimization loop
            for iteration in range(n_iterations):
                # Fit GP model
                self._fit_gp_model()
                
                # Find next point to evaluate
                next_point = self._find_next_point()
                
                # Evaluate objective function
                next_value = objective_func(next_point)
                
                # Update training data
                self.X_train_ = np.vstack([self.X_train_, next_point.reshape(1, -1)])
                self.y_train_ = np.append(self.y_train_, next_value)
                
                # Update best value
                if next_value < self.f_min_:
                    self.f_min_ = next_value
                
                # Re-optimize hyperparameters periodically
                if iteration % 5 == 0:
                    self._optimize_hyperparameters()
            
            # Find best point
            best_idx = np.argmin(self.y_train_)
            best_point = self.X_train_[best_idx]
            best_value = self.y_train_[best_idx]
            
            return best_point, best_value
        
        def _random_points(self, n_points: int) -> np.ndarray:
            """Generate random points within bounds."""
            if self.bounds is None:
                # Default bounds
                bounds = [(-5, 5)] * 2  # 2D default
            else:
                bounds = self.bounds
            
            n_dims = len(bounds)
            points = np.zeros((n_points, n_dims))
            
            for i, (low, high) in enumerate(bounds):
                points[:, i] = np.random.uniform(low, high, n_points)
            
            return points
        
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
                    result = minimize(objective, theta0, bounds=bounds, method='L-BFGS-B')
                    if result.fun < -best_lml:
                        best_theta = result.x
                        best_lml = -result.fun
                except:
                    continue
            
            self.kernel_ = self.kernel.clone_with_theta(best_theta)
        
        def _fit_gp_model(self):
            """Fit the Gaussian Process model."""
            # Compute kernel matrix
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6  # Add noise
            
            # Cholesky decomposition
            self.L_ = cholesky(K, lower=True)
            
            # Solve for alpha
            self.alpha_ = solve_triangular(self.L_, self.y_train_, lower=True)
            self.alpha_ = solve_triangular(self.L_.T, self.alpha_, lower=False)
        
        def _find_next_point(self) -> np.ndarray:
            """Find the next point to evaluate using acquisition function."""
            if self.acquisition_func == 'EI':
                return self._expected_improvement()
            elif self.acquisition_func == 'UCB':
                return self._upper_confidence_bound()
            elif self.acquisition_func == 'PI':
                return self._probability_of_improvement()
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")
        
        def _expected_improvement(self) -> np.ndarray:
            """Find point with maximum expected improvement."""
            def acquisition(x):
                x = x.reshape(1, -1)
                mean, std = self._predict(x)
                
                if std == 0:
                    return 0
                
                improvement = self.f_min_ - mean
                z = improvement / std
                
                # Expected improvement
                ei = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)
                return -ei  # Minimize negative EI
            
            return self._optimize_acquisition(acquisition)
        
        def _upper_confidence_bound(self) -> np.ndarray:
            """Find point with maximum upper confidence bound."""
            def acquisition(x):
                x = x.reshape(1, -1)
                mean, std = self._predict(x)
                return -(mean + self.beta * std)  # Minimize negative UCB
            
            return self._optimize_acquisition(acquisition)
        
        def _probability_of_improvement(self) -> np.ndarray:
            """Find point with maximum probability of improvement."""
            def acquisition(x):
                x = x.reshape(1, -1)
                mean, std = self._predict(x)
                
                if std == 0:
                    return 0
                
                z = (self.f_min_ - mean) / std
                return -self._normal_cdf(z)  # Minimize negative PI
            
            return self._optimize_acquisition(acquisition)
        
        def _optimize_acquisition(self, acquisition_func: Callable) -> np.ndarray:
            """Optimize the acquisition function."""
            best_point = None
            best_value = float('inf')
            
            # Multiple random restarts
            for _ in range(self.n_restarts):
                # Random starting point
                if self.bounds is None:
                    x0 = np.random.uniform(-5, 5, 2)
                else:
                    x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
                
                try:
                    result = minimize(acquisition_func, x0, method='L-BFGS-B', bounds=self.bounds)
                    if result.fun < best_value:
                        best_value = result.fun
                        best_point = result.x
                except:
                    continue
            
            return best_point
        
        def _predict(self, X: np.ndarray) -> Tuple[float, float]:
            """Predict mean and standard deviation at given points."""
            # Compute kernel matrices
            K_star = self.kernel_(self.X_train_, X)
            K_star_star = self.kernel_(X)
            
            # Mean prediction
            mean = K_star.T @ self.alpha_
            
            # Variance prediction
            v = solve_triangular(self.L_, K_star, lower=True)
            var = K_star_star - v.T @ v
            var = np.maximum(var, 1e-10)  # Ensure positive variance
            
            return mean[0], np.sqrt(var[0, 0])
        
        def _log_marginal_likelihood(self) -> float:
            """Compute log marginal likelihood."""
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6
            
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
        
        def _normal_cdf(self, x: float) -> float:
            """Standard normal cumulative distribution function."""
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        def _normal_pdf(self, x: float) -> float:
            """Standard normal probability density function."""
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    ```

=== "Multi-objective GP Optimization"
    ```python
    class MultiObjectiveGPOptimizer:
        """
        Multi-objective Gaussian Process Optimization using Pareto optimization.
        """
        
        def __init__(self, kernel=None, acquisition_func='EI', beta=2.0, 
                     n_restarts=10, bounds=None):
            self.kernel = kernel
            self.acquisition_func = acquisition_func
            self.beta = beta
            self.n_restarts = n_restarts
            self.bounds = bounds
            self.X_train_ = None
            self.y_train_ = None
            self.kernels_ = []
            self.Ls_ = []
            self.alphas_ = []
            self.pareto_front_ = None
        
        def optimize(self, objective_funcs: List[Callable], n_iterations: int, 
                    X_init: Optional[np.ndarray] = None, 
                    y_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            """
            Optimize multiple objective functions simultaneously.
            
            Args:
                objective_funcs: List of functions to optimize
                n_iterations: Number of optimization iterations
                X_init: Initial training points
                y_init: Initial function values
                
            Returns:
                Tuple of (pareto_points, pareto_values)
            """
            n_objectives = len(objective_funcs)
            
            # Initialize with random points if not provided
            if X_init is None:
                X_init = self._random_points(5)
                y_init = np.array([[func(x) for func in objective_funcs] for x in X_init])
            
            self.X_train_ = X_init.copy()
            self.y_train_ = y_init.copy()
            
            # Initialize kernels for each objective
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)
            
            self.kernels_ = [self.kernel.clone() for _ in range(n_objectives)]
            
            # Main optimization loop
            for iteration in range(n_iterations):
                # Fit GP models for each objective
                self._fit_gp_models()
                
                # Find next point to evaluate
                next_point = self._find_next_point_multi_objective()
                
                # Evaluate all objective functions
                next_values = np.array([func(next_point) for func in objective_funcs])
                
                # Update training data
                self.X_train_ = np.vstack([self.X_train_, next_point.reshape(1, -1)])
                self.y_train_ = np.vstack([self.y_train_, next_values.reshape(1, -1)])
                
                # Update Pareto front
                self._update_pareto_front()
                
                # Re-optimize hyperparameters periodically
                if iteration % 5 == 0:
                    self._optimize_hyperparameters()
            
            return self.pareto_front_
        
        def _fit_gp_models(self):
            """Fit GP models for each objective."""
            self.Ls_ = []
            self.alphas_ = []
            
            for i in range(self.y_train_.shape[1]):
                # Compute kernel matrix
                K = self.kernels_[i](self.X_train_)
                K[np.diag_indices_from(K)] += 1e-6
                
                # Cholesky decomposition
                L = cholesky(K, lower=True)
                self.Ls_.append(L)
                
                # Solve for alpha
                alpha = solve_triangular(L, self.y_train_[:, i], lower=True)
                alpha = solve_triangular(L.T, alpha, lower=False)
                self.alphas_.append(alpha)
        
        def _find_next_point_multi_objective(self) -> np.ndarray:
            """Find next point using multi-objective acquisition function."""
            def acquisition(x):
                x = x.reshape(1, -1)
                
                # Predict for all objectives
                means = []
                stds = []
                
                for i in range(len(self.kernels_)):
                    mean, std = self._predict_objective(x, i)
                    means.append(mean)
                    stds.append(std)
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Multi-objective acquisition (simplified)
                # Could use more sophisticated methods like EHVI
                return -np.sum(means + self.beta * stds)
            
            return self._optimize_acquisition(acquisition)
        
        def _predict_objective(self, X: np.ndarray, obj_idx: int) -> Tuple[float, float]:
            """Predict mean and standard deviation for a specific objective."""
            # Compute kernel matrices
            K_star = self.kernels_[obj_idx](self.X_train_, X)
            K_star_star = self.kernels_[obj_idx](X)
            
            # Mean prediction
            mean = K_star.T @ self.alphas_[obj_idx]
            
            # Variance prediction
            v = solve_triangular(self.Ls_[obj_idx], K_star, lower=True)
            var = K_star_star - v.T @ v
            var = np.maximum(var, 1e-10)
            
            return mean[0], np.sqrt(var[0, 0])
        
        def _update_pareto_front(self):
            """Update the Pareto front with new points."""
            # Find Pareto optimal points
            pareto_mask = self._is_pareto_optimal(self.y_train_)
            self.pareto_front_ = (self.X_train_[pareto_mask], self.y_train_[pareto_mask])
        
        def _is_pareto_optimal(self, points: np.ndarray) -> np.ndarray:
            """Check which points are Pareto optimal."""
            n_points = points.shape[0]
            is_pareto = np.ones(n_points, dtype=bool)
            
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        # Check if point j dominates point i
                        if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                            is_pareto[i] = False
                            break
            
            return is_pareto
        
        def _random_points(self, n_points: int) -> np.ndarray:
            """Generate random points within bounds."""
            if self.bounds is None:
                bounds = [(-5, 5)] * 2
            else:
                bounds = self.bounds
            
            n_dims = len(bounds)
            points = np.zeros((n_points, n_dims))
            
            for i, (low, high) in enumerate(bounds):
                points[:, i] = np.random.uniform(low, high, n_points)
            
            return points
        
        def _optimize_acquisition(self, acquisition_func: Callable) -> np.ndarray:
            """Optimize the acquisition function."""
            best_point = None
            best_value = float('inf')
            
            for _ in range(self.n_restarts):
                if self.bounds is None:
                    x0 = np.random.uniform(-5, 5, 2)
                else:
                    x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
                
                try:
                    result = minimize(acquisition_func, x0, method='L-BFGS-B', bounds=self.bounds)
                    if result.fun < best_value:
                        best_value = result.fun
                        best_point = result.x
                except:
                    continue
            
            return best_point
        
        def _optimize_hyperparameters(self):
            """Optimize hyperparameters for all objectives."""
            for i in range(len(self.kernels_)):
                def objective(theta):
                    self.kernels_[i].theta = theta
                    try:
                        return -self._log_marginal_likelihood_objective(i)
                    except np.linalg.LinAlgError:
                        return 1e25
                
                theta0 = self.kernels_[i].theta.copy()
                bounds = self.kernels_[i].bounds
                
                try:
                    result = minimize(objective, theta0, bounds=bounds, method='L-BFGS-B')
                    self.kernels_[i].theta = result.x
                except:
                    continue
        
        def _log_marginal_likelihood_objective(self, obj_idx: int) -> float:
            """Compute log marginal likelihood for a specific objective."""
            K = self.kernels_[obj_idx](self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6
            
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return -1e25
            
            alpha = solve_triangular(L, self.y_train_[:, obj_idx], lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)
            
            data_fit = 0.5 * alpha.T @ self.y_train_[:, obj_idx]
            complexity_penalty = np.sum(np.log(np.diag(L)))
            normalization = 0.5 * len(self.y_train_) * np.log(2 * np.pi)
            
            return -(data_fit + complexity_penalty + normalization)
    ```

=== "Constrained GP Optimization"
    ```python
    class ConstrainedGPOptimizer:
        """
        Gaussian Process Optimization with constraints.
        """
        
        def __init__(self, kernel=None, acquisition_func='EI', beta=2.0, 
                     n_restarts=10, bounds=None):
            self.kernel = kernel
            self.acquisition_func = acquisition_func
            self.beta = beta
            self.n_restarts = n_restarts
            self.bounds = bounds
            self.X_train_ = None
            self.y_train_ = None
            self.c_train_ = None  # Constraint values
            self.kernel_ = None
            self.constraint_kernels_ = []
            self.L_ = None
            self.alpha_ = None
            self.f_min_ = None
        
        def optimize(self, objective_func: Callable, constraint_funcs: List[Callable], 
                    n_iterations: int, X_init: Optional[np.ndarray] = None, 
                    y_init: Optional[np.ndarray] = None, 
                    c_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
            """
            Optimize objective function subject to constraints.
            
            Args:
                objective_func: Function to optimize
                constraint_funcs: List of constraint functions (should be <= 0)
                n_iterations: Number of optimization iterations
                X_init: Initial training points
                y_init: Initial function values
                c_init: Initial constraint values
                
            Returns:
                Tuple of (best_point, best_value)
            """
            n_constraints = len(constraint_funcs)
            
            # Initialize with random points if not provided
            if X_init is None:
                X_init = self._random_points(5)
                y_init = np.array([objective_func(x) for x in X_init])
                c_init = np.array([[func(x) for func in constraint_funcs] for x in X_init])
            
            self.X_train_ = X_init.copy()
            self.y_train_ = y_init.copy()
            self.c_train_ = c_init.copy()
            
            # Find feasible points
            feasible_mask = np.all(self.c_train_ <= 0, axis=1)
            if not np.any(feasible_mask):
                raise ValueError("No feasible points in initial set")
            
            self.f_min_ = np.min(self.y_train_[feasible_mask])
            
            # Initialize kernels
            if self.kernel is None:
                from sklearn.gaussian_process.kernels import RBF
                self.kernel = RBF(1.0)
            
            self.kernel_ = self.kernel.clone()
            self.constraint_kernels_ = [self.kernel.clone() for _ in range(n_constraints)]
            
            # Main optimization loop
            for iteration in range(n_iterations):
                # Fit GP models
                self._fit_gp_models()
                
                # Find next point to evaluate
                next_point = self._find_next_point_constrained()
                
                # Evaluate objective and constraints
                next_value = objective_func(next_point)
                next_constraints = np.array([func(next_point) for func in constraint_funcs])
                
                # Update training data
                self.X_train_ = np.vstack([self.X_train_, next_point.reshape(1, -1)])
                self.y_train_ = np.append(self.y_train_, next_value)
                self.c_train_ = np.vstack([self.c_train_, next_constraints.reshape(1, -1)])
                
                # Update best value if feasible
                if np.all(next_constraints <= 0) and next_value < self.f_min_:
                    self.f_min_ = next_value
                
                # Re-optimize hyperparameters periodically
                if iteration % 5 == 0:
                    self._optimize_hyperparameters()
            
            # Find best feasible point
            feasible_mask = np.all(self.c_train_ <= 0, axis=1)
            if not np.any(feasible_mask):
                raise ValueError("No feasible points found")
            
            best_idx = np.argmin(self.y_train_[feasible_mask])
            feasible_indices = np.where(feasible_mask)[0]
            best_point = self.X_train_[feasible_indices[best_idx]]
            best_value = self.y_train_[feasible_indices[best_idx]]
            
            return best_point, best_value
        
        def _fit_gp_models(self):
            """Fit GP models for objective and constraints."""
            # Fit objective model
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6
            self.L_ = cholesky(K, lower=True)
            self.alpha_ = solve_triangular(self.L_, self.y_train_, lower=True)
            self.alpha_ = solve_triangular(self.L_.T, self.alpha_, lower=False)
            
            # Fit constraint models
            self.constraint_Ls_ = []
            self.constraint_alphas_ = []
            
            for i in range(self.c_train_.shape[1]):
                K_c = self.constraint_kernels_[i](self.X_train_)
                K_c[np.diag_indices_from(K_c)] += 1e-6
                
                L_c = cholesky(K_c, lower=True)
                self.constraint_Ls_.append(L_c)
                
                alpha_c = solve_triangular(L_c, self.c_train_[:, i], lower=True)
                alpha_c = solve_triangular(L_c.T, alpha_c, lower=False)
                self.constraint_alphas_.append(alpha_c)
        
        def _find_next_point_constrained(self) -> np.ndarray:
            """Find next point using constrained acquisition function."""
            def acquisition(x):
                x = x.reshape(1, -1)
                
                # Predict objective
                obj_mean, obj_std = self._predict_objective(x)
                
                # Predict constraints
                constraint_means = []
                constraint_stds = []
                
                for i in range(len(self.constraint_kernels_)):
                    mean, std = self._predict_constraint(x, i)
                    constraint_means.append(mean)
                    constraint_stds.append(std)
                
                constraint_means = np.array(constraint_means)
                constraint_stds = np.array(constraint_stds)
                
                # Constrained acquisition function
                # Penalize constraint violations
                constraint_penalty = 0
                for i, (mean, std) in enumerate(zip(constraint_means, constraint_stds)):
                    # Probability of constraint violation
                    prob_violation = self._normal_cdf(mean / std) if std > 0 else 0
                    constraint_penalty += prob_violation * 1000  # Large penalty
                
                # Expected improvement with constraint penalty
                if obj_std == 0:
                    ei = 0
                else:
                    improvement = self.f_min_ - obj_mean
                    z = improvement / obj_std
                    ei = improvement * self._normal_cdf(z) + obj_std * self._normal_pdf(z)
                
                return -(ei - constraint_penalty)
            
            return self._optimize_acquisition(acquisition)
        
        def _predict_objective(self, X: np.ndarray) -> Tuple[float, float]:
            """Predict objective function mean and standard deviation."""
            K_star = self.kernel_(self.X_train_, X)
            K_star_star = self.kernel_(X)
            
            mean = K_star.T @ self.alpha_
            
            v = solve_triangular(self.L_, K_star, lower=True)
            var = K_star_star - v.T @ v
            var = np.maximum(var, 1e-10)
            
            return mean[0], np.sqrt(var[0, 0])
        
        def _predict_constraint(self, X: np.ndarray, constraint_idx: int) -> Tuple[float, float]:
            """Predict constraint function mean and standard deviation."""
            K_star = self.constraint_kernels_[constraint_idx](self.X_train_, X)
            K_star_star = self.constraint_kernels_[constraint_idx](X)
            
            mean = K_star.T @ self.constraint_alphas_[constraint_idx]
            
            v = solve_triangular(self.constraint_Ls_[constraint_idx], K_star, lower=True)
            var = K_star_star - v.T @ v
            var = np.maximum(var, 1e-10)
            
            return mean[0], np.sqrt(var[0, 0])
        
        def _random_points(self, n_points: int) -> np.ndarray:
            """Generate random points within bounds."""
            if self.bounds is None:
                bounds = [(-5, 5)] * 2
            else:
                bounds = self.bounds
            
            n_dims = len(bounds)
            points = np.zeros((n_points, n_dims))
            
            for i, (low, high) in enumerate(bounds):
                points[:, i] = np.random.uniform(low, high, n_points)
            
            return points
        
        def _optimize_acquisition(self, acquisition_func: Callable) -> np.ndarray:
            """Optimize the acquisition function."""
            best_point = None
            best_value = float('inf')
            
            for _ in range(self.n_restarts):
                if self.bounds is None:
                    x0 = np.random.uniform(-5, 5, 2)
                else:
                    x0 = np.array([np.random.uniform(low, high) for low, high in self.bounds])
                
                try:
                    result = minimize(acquisition_func, x0, method='L-BFGS-B', bounds=self.bounds)
                    if result.fun < best_value:
                        best_value = result.fun
                        best_point = result.x
                except:
                    continue
            
            return best_point
        
        def _normal_cdf(self, x: float) -> float:
            """Standard normal cumulative distribution function."""
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        def _normal_pdf(self, x: float) -> float:
            """Standard normal probability density function."""
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
        def _optimize_hyperparameters(self):
            """Optimize hyperparameters for all models."""
            # Optimize objective kernel
            def objective_theta(theta):
                self.kernel_.theta = theta
                try:
                    return -self._log_marginal_likelihood_objective()
                except np.linalg.LinAlgError:
                    return 1e25
            
            theta0 = self.kernel_.theta.copy()
            bounds = self.kernel_.bounds
            
            try:
                result = minimize(objective_theta, theta0, bounds=bounds, method='L-BFGS-B')
                self.kernel_.theta = result.x
            except:
                pass
            
            # Optimize constraint kernels
            for i in range(len(self.constraint_kernels_)):
                def constraint_theta(theta):
                    self.constraint_kernels_[i].theta = theta
                    try:
                        return -self._log_marginal_likelihood_constraint(i)
                    except np.linalg.LinAlgError:
                        return 1e25
                
                theta0 = self.constraint_kernels_[i].theta.copy()
                bounds = self.constraint_kernels_[i].bounds
                
                try:
                    result = minimize(constraint_theta, theta0, bounds=bounds, method='L-BFGS-B')
                    self.constraint_kernels_[i].theta = result.x
                except:
                    continue
        
        def _log_marginal_likelihood_objective(self) -> float:
            """Compute log marginal likelihood for objective."""
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6
            
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return -1e25
            
            alpha = solve_triangular(L, self.y_train_, lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)
            
            data_fit = 0.5 * alpha.T @ self.y_train_
            complexity_penalty = np.sum(np.log(np.diag(L)))
            normalization = 0.5 * len(self.y_train_) * np.log(2 * np.pi)
            
            return -(data_fit + complexity_penalty + normalization)
        
        def _log_marginal_likelihood_constraint(self, constraint_idx: int) -> float:
            """Compute log marginal likelihood for constraint."""
            K = self.constraint_kernels_[constraint_idx](self.X_train_)
            K[np.diag_indices_from(K)] += 1e-6
            
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return -1e25
            
            alpha = solve_triangular(L, self.c_train_[:, constraint_idx], lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)
            
            data_fit = 0.5 * alpha.T @ self.c_train_[:, constraint_idx]
            complexity_penalty = np.sum(np.log(np.diag(L)))
            normalization = 0.5 * len(self.c_train_) * np.log(2 * np.pi)
            
            return -(data_fit + complexity_penalty + normalization)
    ```

!!! tip "Complete Implementation"
    The full implementation with comprehensive testing, additional variants, and performance optimizations is available in the source code:

    - **Main Implementation**: [`src/algokit/gaussian_process/gp_optimization.py`](https://github.com/jeffrichley/algokit/blob/main/src/algokit/gaussian_process/gp_optimization.py)
    - **Tests**: [`tests/unit/gaussian_process/test_gp_optimization.py`](https://github.com/jeffrichley/algokit/blob/main/tests/unit/gaussian_process/test_gp_optimization.py)

## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    **Standard GPO** | $O(n^3 + n \cdot d)$ | $O(n^2)$ | n evaluations, d dimensions |
    **Multi-objective GPO** | $O(m \cdot n^3 + n \cdot d)$ | $O(m \cdot n^2)$ | m objectives |
    **Constrained GPO** | $O((c+1) \cdot n^3 + n \cdot d)$ | $O((c+1) \cdot n^2)$ | c constraints |

!!! warning "Performance Considerations"
    - **Cubic complexity** in number of evaluations limits scalability
    - **Acquisition function optimization** can be computationally expensive
    - **Hyperparameter optimization** requires multiple restarts
    - **Memory usage** grows quadratically with evaluations

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Hyperparameter Optimization"
        - **Machine Learning**: Neural network architecture search
        - **Model Tuning**: Parameter optimization for ML models
        - **Feature Selection**: Optimal feature subset selection
        - **Algorithm Configuration**: Automated algorithm tuning

    !!! grid-item "Engineering Design"
        - **Aerospace**: Aircraft wing design optimization
        - **Automotive**: Engine parameter optimization
        - **Materials**: Composite material design
        - **Electronics**: Circuit design optimization

    !!! grid-item "Scientific Research"
        - **Drug Discovery**: Molecular property optimization
        - **Climate Modeling**: Parameter estimation
        - **Physics**: Experimental design optimization
        - **Chemistry**: Reaction condition optimization

    !!! grid-item "Educational Value"
        - **Bayesian Optimization**: Understanding acquisition functions
        - **Global Optimization**: Learning efficient search strategies
        - **Uncertainty Quantification**: Understanding exploration vs exploitation
        - **Kernel Methods**: Learning GP-based optimization

!!! success "Educational Value"
    - **Bayesian Optimization**: Perfect example of intelligent search strategies
    - **Acquisition Functions**: Shows how to balance exploration and exploitation
    - **Global Optimization**: Demonstrates efficient global search methods
    - **Uncertainty Quantification**: Illustrates how to use uncertainty in decision making

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Core Papers"
        1. **Mockus, J.** (1974). On Bayesian methods for seeking the extremum. *Optimization Techniques IFIP Technical Conference*, 400-404.
        2. **Jones, D. R., Schonlau, M., & Welch, W. J.** (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

    !!! grid-item "Bayesian Optimization Textbooks"
        3. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
        4. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N.** (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.

    !!! grid-item "Online Resources"
        5. [Bayesian Optimization - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_optimization)
        6. [GPyOpt Library](https://github.com/SheffieldML/GPyOpt) - Python Bayesian optimization library
        7. [BoTorch Library](https://github.com/pytorch/botorch) - PyTorch-based Bayesian optimization

    !!! grid-item "Implementation & Practice"
        8. [scikit-optimize Library](https://github.com/scikit-optimize/scikit-optimize) - Scikit-learn compatible optimization
        9. [Optuna Library](https://github.com/optuna/optuna) - Hyperparameter optimization framework
        10. [Bayesian Optimization Tutorial](https://distill.pub/2020/bayesian-optimization/)

!!! tip "Interactive Learning"
    Try implementing GP Optimization yourself! Start with simple 1D and 2D functions to understand how the algorithm works. Experiment with different acquisition functions to see how they affect the search behavior. Try implementing multi-objective optimization to understand Pareto optimization. Compare GP optimization with other global optimization methods to see the benefits of the Bayesian approach. This will give you deep insight into intelligent optimization and acquisition function design.

## Navigation

{{ nav_grid(current_algorithm="gp-optimization", current_family="gaussian-process", max_related=5) }}
