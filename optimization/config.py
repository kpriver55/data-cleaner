"""
Configuration for DSPy Optimization

Manages settings and hyperparameters for optimization runs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import yaml


@dataclass
class OptimizerConfig:
    """
    Configuration for DSPy optimizer

    Attributes:
        optimizer_type: Type of optimizer ('BootstrapFewShot', 'MIPRO', etc.)
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        max_labeled_demos: Maximum number of labeled demonstrations to use
        num_candidate_programs: Number of candidate programs to generate (MIPRO)
        num_threads: Number of threads for parallel optimization
        max_errors: Maximum number of errors before stopping
        metric_threshold: Minimum metric score to consider successful (for binary metrics)
        teacher_settings: Optional settings for teacher model (if different from student)
    """
    optimizer_type: str = "BootstrapFewShot"
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_candidate_programs: int = 10
    num_threads: int = 8
    max_errors: int = 5
    metric_threshold: float = 0.7
    teacher_settings: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration"""
        valid_optimizers = [
            'BootstrapFewShot',
            'BootstrapFewShotWithRandomSearch',
            'MIPRO',
            'MIPROv2'
        ]
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"optimizer_type must be one of {valid_optimizers}, "
                f"got '{self.optimizer_type}'"
            )

        if self.max_bootstrapped_demos < 1:
            raise ValueError("max_bootstrapped_demos must be >= 1")

        if self.max_labeled_demos < 1:
            raise ValueError("max_labeled_demos must be >= 1")

        if not 0 < self.metric_threshold <= 1:
            raise ValueError("metric_threshold must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'optimizer_type': self.optimizer_type,
            'max_bootstrapped_demos': self.max_bootstrapped_demos,
            'max_labeled_demos': self.max_labeled_demos,
            'num_candidate_programs': self.num_candidate_programs,
            'num_threads': self.num_threads,
            'max_errors': self.max_errors,
            'metric_threshold': self.metric_threshold,
            'teacher_settings': self.teacher_settings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizerConfig':
        """Create from dictionary"""
        return cls(**data)

    def save(self, path: str):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'OptimizerConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation

    Attributes:
        evaluator_type: Type of evaluator ('default', 'custom')
        weights: Custom weights for composite evaluator components
        use_binary_metric: Whether to use binary (pass/fail) metric
        binary_threshold: Threshold for binary metric
    """
    evaluator_type: str = "default"
    weights: Optional[Dict[str, float]] = None
    use_binary_metric: bool = False
    binary_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'evaluator_type': self.evaluator_type,
            'weights': self.weights,
            'use_binary_metric': self.use_binary_metric,
            'binary_threshold': self.binary_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class OptimizationConfig:
    """
    Complete configuration for an optimization run

    Attributes:
        name: Name/identifier for this optimization run
        description: Description of the optimization goal
        dataset_path: Path to training dataset (JSON or YAML)
        output_dir: Directory to save optimization results
        optimizer: Optimizer configuration
        evaluation: Evaluation configuration
        train_ratio: Ratio of data to use for training (rest for validation)
        random_seed: Random seed for reproducibility
    """
    name: str
    dataset_path: str
    output_dir: str = "optimization_results"
    description: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    train_ratio: float = 0.8
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration"""
        if not 0 < self.train_ratio <= 1:
            raise ValueError("train_ratio must be between 0 and 1")

        if not Path(self.dataset_path).exists():
            raise ValueError(f"dataset_path does not exist: {self.dataset_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'optimizer': self.optimizer.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'train_ratio': self.train_ratio,
            'random_seed': self.random_seed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary"""
        # Handle nested configs
        if 'optimizer' in data and isinstance(data['optimizer'], dict):
            data['optimizer'] = OptimizerConfig.from_dict(data['optimizer'])
        if 'evaluation' in data and isinstance(data['evaluation'], dict):
            data['evaluation'] = EvaluationConfig.from_dict(data['evaluation'])
        return cls(**data)

    def save_json(self, path: str):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_json(cls, path: str) -> 'OptimizationConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_yaml(cls, path: str) -> 'OptimizationConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def get_output_path(self, filename: str) -> Path:
        """
        Get path for an output file within the output directory

        Args:
            filename: Name of the output file

        Returns:
            Path object for the output file
        """
        output_dir = Path(self.output_dir) / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename


# Predefined configurations for common use cases

def create_quick_optimization_config(
    name: str,
    dataset_path: str,
    description: Optional[str] = None
) -> OptimizationConfig:
    """
    Create a configuration for quick optimization (few examples, fast)

    Good for: Initial testing, small datasets

    Args:
        name: Name for the optimization run
        dataset_path: Path to training dataset
        description: Optional description

    Returns:
        OptimizationConfig with quick settings
    """
    return OptimizationConfig(
        name=name,
        dataset_path=dataset_path,
        description=description or "Quick optimization with BootstrapFewShot",
        optimizer=OptimizerConfig(
            optimizer_type="BootstrapFewShot",
            max_bootstrapped_demos=3,
            max_labeled_demos=8,
            num_threads=4
        ),
        train_ratio=0.8
    )


def create_thorough_optimization_config(
    name: str,
    dataset_path: str,
    description: Optional[str] = None
) -> OptimizationConfig:
    """
    Create a configuration for thorough optimization (more examples, slower)

    Good for: Production models, large datasets

    Args:
        name: Name for the optimization run
        dataset_path: Path to training dataset
        description: Optional description

    Returns:
        OptimizationConfig with thorough settings
    """
    return OptimizationConfig(
        name=name,
        dataset_path=dataset_path,
        description=description or "Thorough optimization with MIPRO",
        optimizer=OptimizerConfig(
            optimizer_type="MIPRO",
            max_bootstrapped_demos=8,
            max_labeled_demos=16,
            num_candidate_programs=20,
            num_threads=8
        ),
        train_ratio=0.85
    )


def create_balanced_optimization_config(
    name: str,
    dataset_path: str,
    description: Optional[str] = None
) -> OptimizationConfig:
    """
    Create a configuration with balanced settings

    Good for: Most use cases, reasonable speed and quality

    Args:
        name: Name for the optimization run
        dataset_path: Path to training dataset
        description: Optional description

    Returns:
        OptimizationConfig with balanced settings
    """
    return OptimizationConfig(
        name=name,
        dataset_path=dataset_path,
        description=description or "Balanced optimization with BootstrapFewShot",
        optimizer=OptimizerConfig(
            optimizer_type="BootstrapFewShot",
            max_bootstrapped_demos=4,
            max_labeled_demos=12,
            num_threads=8
        ),
        train_ratio=0.8
    )
