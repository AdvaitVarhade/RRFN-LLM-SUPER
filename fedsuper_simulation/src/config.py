"""
fedsuper_simulation/src/config.py - Simulation Configuration Dataclass
Encapsulates all simulation hyperparameters, DP settings, SUPER calibration parameters, and validation.
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import math


@dataclass
class SimulationConfig:
    """Configuration container for Privacy-Preserving Federated SUPER simulation.

    Attributes:
        num_clients: Total number of federated client devices (users).
        num_items: Total number of items in the recommendation catalog.
        embedding_dim: Latent dimension for collaborative filtering embeddings (p_u, q_i).
        llm_dim: Dimension for semantic LLM profile and item text vectors (s_u, v_i).
        pareto_alpha: Cumulative interaction volume threshold for Head items (default 0.20).
        calibration_alpha: Popularity calibration blend strength alpha in [0, 1] (0.0=uncalibrated, 1.0=strict blueprint).
        llm_lambda: Semantic score blend weight lambda in [0, 1] (0.0=pure CF, 1.0=pure LLM semantic cosine).
        dp_enabled: Whether Differential Privacy clipping and Gaussian noise are active.
        dp_epsilon: Privacy budget epsilon (lower = stronger privacy, higher noise).
        dp_delta: Privacy parameter delta (probability of privacy breach, default 1e-5).
        dp_l2_clip_norm: Maximum L2 norm clipping threshold C for client item gradient updates.
        top_k: Length of recommendation slate presented to each client (Top-K).
        clients_per_round: Number of clients sampled per federated communication round.
        learning_rate: Local client SGD learning rate for BPR matrix factorization.
        max_rounds: Total number of federated rounds to execute.
        seed: Master random seed for deterministic reproducibility across simulation runs.
        merge_mode: Quota allocation strategy ('stochastic', 'static', 'adaptive_alpha', 'confidence_elastic', 'hybrid').
        num_genres: Number of synthetic genre categories (default 6).
        zipf_exponent: Power-law decay exponent gamma for item popularity distribution (default 1.25).
    """
    num_clients: int = 20
    num_items: int = 100
    embedding_dim: int = 32
    llm_dim: int = 32
    pareto_alpha: float = 0.20
    calibration_alpha: float = 0.40
    llm_lambda: float = 0.70
    dp_enabled: bool = True
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    dp_l2_clip_norm: float = 1.0
    top_k: int = 10
    clients_per_round: int = 5
    learning_rate: float = 0.05
    max_rounds: int = 20
    seed: int = 42
    merge_mode: str = "stochastic"
    num_genres: int = 6
    zipf_exponent: float = 1.25

    def __post_init__(self) -> None:
        """Validates all hyperparameter ranges and consistency constraints."""
        if self.num_clients <= 0:
            raise ValueError(f"num_clients must be > 0, got {self.num_clients}")
        if self.num_items <= 0:
            raise ValueError(f"num_items must be > 0, got {self.num_items}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.llm_dim <= 0:
            raise ValueError(f"llm_dim must be > 0, got {self.llm_dim}")
        if not (0.0 <= self.pareto_alpha <= 1.0):
            raise ValueError(f"pareto_alpha must be in [0.0, 1.0], got {self.pareto_alpha}")
        if not (0.0 <= self.calibration_alpha <= 1.0):
            raise ValueError(f"calibration_alpha must be in [0.0, 1.0], got {self.calibration_alpha}")
        if not (0.0 <= self.llm_lambda <= 1.0):
            raise ValueError(f"llm_lambda must be in [0.0, 1.0], got {self.llm_lambda}")
        if self.dp_epsilon <= 0.0:
            raise ValueError(f"dp_epsilon must be > 0, got {self.dp_epsilon}")
        if not (0.0 < self.dp_delta < 1.0):
            raise ValueError(f"dp_delta must be in (0.0, 1.0), got {self.dp_delta}")
        if self.dp_l2_clip_norm <= 0.0:
            raise ValueError(f"dp_l2_clip_norm must be > 0, got {self.dp_l2_clip_norm}")
        if self.top_k <= 0 or self.top_k > self.num_items:
            raise ValueError(f"top_k must be in 1..num_items ({self.num_items}), got {self.top_k}")
        if self.clients_per_round <= 0 or self.clients_per_round > self.num_clients:
            raise ValueError(f"clients_per_round must be in 1..num_clients ({self.num_clients}), got {self.clients_per_round}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_rounds <= 0:
            raise ValueError(f"max_rounds must be > 0, got {self.max_rounds}")
        valid_merge_modes = {"stochastic", "static", "adaptive_alpha", "confidence_elastic", "hybrid"}
        if self.merge_mode not in valid_merge_modes:
            raise ValueError(f"merge_mode '{self.merge_mode}' invalid; must be one of {valid_merge_modes}")
        if self.num_genres <= 0:
            raise ValueError(f"num_genres must be > 0, got {self.num_genres}")

    @property
    def dp_sigma(self) -> float:
        """Computes the theoretical Gaussian DP noise standard deviation sigma.

        Formula:
            sigma = (C * sqrt(2 * ln(1.25 / delta))) / epsilon
        """
        if not self.dp_enabled or self.dp_epsilon <= 0.0:
            return 0.0
        return (self.dp_l2_clip_norm * math.sqrt(2.0 * math.log(1.25 / self.dp_delta))) / self.dp_epsilon

    def to_dict(self) -> Dict[str, Any]:
        """Serializes config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Instantiates config from dictionary, ignoring extraneous keys."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
