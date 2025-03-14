import torch
from typing import Tuple
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from tqdm import tqdm
from dldna.chapter_04.models.base import SimpleNetwork  # Import SimpleNetwork
from dldna.chapter_04.utils.data import get_data_loaders, get_device
import torch.nn as nn
import torch.optim as optim
from dldna.chapter_04.experiments.model_training import train_model


class HyperTuner:
    """Bayesian optimization-based tuner for neural network hyperparameters."""

    def __init__(self, max_trials: int = 80, init_samples: int = 10):
        """
        Args:
            max_trials: Number of optimization attempts.
            init_samples: Number of initial random sampling attempts.
        """
        self.max_trials = max_trials
        self.init_samples = init_samples
        self.device = get_device()  # Get the device

        # Parameter bounds (learning rate, batch size, hidden layer 1, hidden layer 2)
        self.param_bounds = torch.tensor([
            [1e-4, 64.0, 64.0, 32.0],      # Minimum values
            [1e-2, 256.0, 512.0, 256.0]    # Maximum values
        ], dtype=torch.float64)  # Use float64 for better numerical stability
        self.num_params = self.param_bounds.shape[1] # dynamic parameter

    def _generate_initial_samples(self):
        """Generates and evaluates initial random configurations."""
        print(f"\nGenerating {self.init_samples} initial samples...")

        # Random sampling within parameter bounds
        raw_configs = self.param_bounds[0] + (
            self.param_bounds[1] - self.param_bounds[0]
        ) * torch.rand(self.init_samples, self.num_params, dtype=torch.float64)

        # Normalize configurations and evaluate model
        norm_configs = normalize(raw_configs, self.param_bounds)
        # Evaluate each configuration and store the resulting accuracy
        accuracies = torch.tensor(
            [self._evaluate_model(params) for params in raw_configs],
            dtype=torch.float64
        )

        return norm_configs, accuracies.unsqueeze(-1) # (num_samples, 1) 형태로.

    def _evaluate_model(self, params: Tensor) -> float:
        """Evaluates the model with the given parameters."""

        # Unpack parameters (and convert to appropriate types)
        learning_rate = float(params[0])
        batch_size = int(params[1])
        hidden1 = int(params[2])
        hidden2 = int(params[3])
        hidden_layers = [hidden1, hidden2]

        # Get data loaders (use CIFAR100, as in the original notebook)
        train_loader, test_loader = get_data_loaders(dataset="CIFAR100", batch_size=batch_size)


        # Instantiate the model
        model = SimpleNetwork(act_func=nn.ReLU(), input_shape=3*32*32,
                              hidden_shape=hidden_layers, num_labels=100).to(self.device)

        # Optimizer (use Adam)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model using the Chapter 4 training function.
        results = train_model(model, train_loader, test_loader, self.device,
                              optimizer=optimizer, epochs=20, retrain=True, save_dir="./tmp/bayes_opt")

        # Return the final test accuracy
        return results['test_accuracies'][-1]


    def tune(self) -> Tuple[Tensor, float]:
      """Performs Bayesian optimization.

      Returns:
          best_config: The best hyperparameter configuration found.
          best_accuracy: The best accuracy achieved.
      """
      # Generate initial samples
      configs, accuracies = self._generate_initial_samples()
      best_accuracy = accuracies.max().item()
      print(f"Initial best accuracy: {best_accuracy:.4f}")

      # Perform sequential optimization
      for trial in tqdm(range(self.max_trials)):
          # Fit Gaussian Process model
          model = SingleTaskGP(configs, accuracies)  # Use the generated data
          mll = ExactMarginalLogLikelihood(model.likelihood, model)
          fit_gpytorch_mll(mll)

          # Create acquisition function (LogEI)
          acq_func = LogExpectedImprovement(
              model, best_f=accuracies.max().item()
          )

          # Define bounds for normalized parameters (0 to 1)
          bounds = torch.stack([
              torch.zeros(self.num_params, dtype=torch.float64),
              torch.ones(self.num_params, dtype=torch.float64)
          ]).to(self.device)

          # Optimize acquisition function to find the next candidate
          candidate, _ = optimize_acqf(
              acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=512
          )

          # Evaluate the new candidate
          raw_candidate = unnormalize(candidate, self.param_bounds)
          accuracy = self._evaluate_model(raw_candidate[0])

          # Update data
          configs = torch.cat([configs, candidate])
          accuracies = torch.cat([
              accuracies,
              torch.tensor([[accuracy]], dtype=torch.float64) # (1,1)
          ])

          # Print progress
          if accuracy > best_accuracy:
              best_accuracy = accuracy
              print(f"\nTrial {trial + 1}: New best accuracy: {accuracy:.4f}")
              print(f"Parameters: learning_rate={raw_candidate[0,0]:.4e}, "
                    f"batch_size={int(raw_candidate[0,1])}, "
                    f"hidden_layers=[{int(raw_candidate[0,2])}, {int(raw_candidate[0,3])}]")

      best_idx = accuracies.argmax()
      best_config = unnormalize(configs[best_idx], self.param_bounds)
      return best_config, accuracies[best_idx]



def run_botorch_optimization(max_trials: int = 40, init_samples: int = 10):
    """
    Main function to run the BoTorch hyperparameter optimization.  This
    function can be called from a Jupyter Notebook.

    Args:
        max_trials: The total number of optimization trials.
        init_samples:  The number of initial random samples.

    """
    print("Starting SimpleNet hyperparameter optimization with BoTorch...")
    tuner = HyperTuner(max_trials=max_trials, init_samples=init_samples)
    best_config, best_acc = tuner.tune()

    print("\nOptimization Complete!")
    print(f"Best Accuracy: {best_acc.item():.4f}")
    print(f"Best Config:")
    print(f"- Learning Rate: {best_config[0]:.4e}")
    print(f"- Batch Size: {int(best_config[1])}")
    print(f"- Hidden Layers: [{int(best_config[2])}, {int(best_config[3])}]")