"""
fedsuper_simulation/src/federated_core.py
Standalone Privacy-Preserving Federated Collaborative Filtering Engine.
Implements local BPR Matrix Factorization, Differential Privacy (L2 gradient clipping and Gaussian noise),
Sparse FedAvg server aggregation, and client privacy enclaves (zero egress of private user representations).
"""
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

try:
    from .config import SimulationConfig
    from .mock_data import SyntheticDataset
    from .super_engine import (
        compute_user_popularity_blueprint,
        fuse_cf_and_llm_scores,
        calibrated_blueprint_merge
    )
except (ImportError, ValueError):
    try:
        from src.config import SimulationConfig
        from src.mock_data import SyntheticDataset
        from src.super_engine import (
            compute_user_popularity_blueprint,
            fuse_cf_and_llm_scores,
            calibrated_blueprint_merge
        )
    except ImportError:
        from config import SimulationConfig
        from mock_data import SyntheticDataset
        from super_engine import (
            compute_user_popularity_blueprint,
            fuse_cf_and_llm_scores,
            calibrated_blueprint_merge
        )


class FederatedServer:
    """
    Central Parameter Hub storing global item embeddings Q and biases b.
    Aggregates parameter deltas from active clients via Sparse FedAvg.
    """
    def __init__(self, num_items: int, embedding_dim: int, seed: int = 42):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed)
        self.item_embeddings = rng.uniform(-0.01, 0.01, size=(num_items, embedding_dim)).astype(np.float32)
        self.item_biases = np.zeros(num_items, dtype=np.float32)

    def aggregate_updates(self, updates: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Sparse FedAvg: updates only items touched by sampled clients in the current round.
        updates: list of (touched_indices, delta_embeddings, delta_biases)
        """
        if not updates:
            return {"items_touched_count": 0, "items_touched_pct": 0.0}

        sum_emb = np.zeros_like(self.item_embeddings)
        cnt_emb = np.zeros(self.num_items, dtype=np.float32)
        sum_bias = np.zeros_like(self.item_biases)
        cnt_bias = np.zeros(self.num_items, dtype=np.float32)

        for touched_idx, delta_emb, delta_bias in updates:
            if len(touched_idx) == 0:
                continue
            sum_emb[touched_idx] += delta_emb
            cnt_emb[touched_idx] += 1.0
            sum_bias[touched_idx] += delta_bias
            cnt_bias[touched_idx] += 1.0

        touched_mask = cnt_emb > 0
        if np.any(touched_mask):
            self.item_embeddings[touched_mask] += sum_emb[touched_mask] / cnt_emb[touched_mask, None]
            self.item_biases[touched_mask] += sum_bias[touched_mask] / cnt_bias[touched_mask]

        touched_count = int(np.sum(touched_mask))
        return {
            "items_touched_count": touched_count,
            "items_touched_pct": float(touched_count / self.num_items) * 100.0
        }


class FederatedClient:
    """
    Local Privacy Enclave on client device.
    Holds private user embedding p_u, interaction history, and LLM semantic profile.
    Executes BPR local training with DP L2 clipping and Gaussian noise.
    """
    def __init__(self, user_id: int, train_items: np.ndarray, llm_profile: np.ndarray,
                 embedding_dim: int, seed: int = 42):
        self.user_id = user_id
        self.train_items = np.asarray(train_items, dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.user_embedding = rng.uniform(-0.01, 0.01, size=embedding_dim).astype(np.float32)
        self.llm_profile = np.asarray(llm_profile, dtype=np.float32)
        norm = np.linalg.norm(self.llm_profile)
        if norm > 1e-12:
            self.llm_profile = (self.llm_profile / norm).astype(np.float32)

    def local_train_step(
        self,
        global_item_emb: np.ndarray,
        global_item_biases: np.ndarray,
        config: SimulationConfig,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Executes local BPR gradient step.
        Returns: (touched_item_indices, delta_item_embeddings, delta_item_biases, mean_loss)
        """
        if len(self.train_items) == 0:
            return np.array([], dtype=np.int64), np.zeros((0, config.embedding_dim), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0

        if rng is None:
            rng = np.random.default_rng()

        num_items = global_item_emb.shape[0]
        lr = float(config.learning_rate)
        clip_c = float(config.dp_l2_clip_norm)
        dp_enabled = bool(config.dp_enabled)

        # Compute DP sigma
        if dp_enabled and config.dp_epsilon > 0:
            sigma_dp = (clip_c * np.sqrt(2.0 * np.log(1.25 / max(config.dp_delta, 1e-15)))) / max(config.dp_epsilon, 1e-5)
        else:
            sigma_dp = 0.0

        # Sample positive interactions
        pos_items = self.train_items
        sweep_size = min(len(pos_items), 8)
        chosen_pos = rng.choice(pos_items, size=sweep_size, replace=False)

        # Build candidate negative set (items not in train_items)
        seen_set = set(pos_items.tolist())
        unseen_items = np.array([it for it in range(num_items) if it not in seen_set], dtype=np.int64)
        if len(unseen_items) == 0:
            return np.array([], dtype=np.int64), np.zeros((0, config.embedding_dim), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0

        chosen_negs = rng.choice(unseen_items, size=sweep_size, replace=True)

        touched_deltas_emb: Dict[int, np.ndarray] = {}
        touched_deltas_bias: Dict[int, float] = {}
        losses = []

        for pos_id, neg_id in zip(chosen_pos, chosen_negs):
            pos_id = int(pos_id)
            neg_id = int(neg_id)

            q_pos = global_item_emb[pos_id]
            b_pos = global_item_biases[pos_id]
            q_neg = global_item_emb[neg_id]
            b_neg = global_item_biases[neg_id]

            # Forward scores
            s_pos = float(np.dot(self.user_embedding, q_pos) + b_pos)
            s_neg = float(np.dot(self.user_embedding, q_neg) + b_neg)
            diff = s_pos - s_neg

            # Numerically stable BPR loss & gamma
            loss = float(np.logaddexp(0.0, -diff))
            losses.append(loss)

            gamma = float(1.0 / (1.0 + np.exp(np.clip(diff, -30.0, 30.0))))

            # Gradients
            # User embedding grad (kept local, strictly private)
            g_p = -gamma * (q_pos - q_neg) + 1e-4 * self.user_embedding
            self.user_embedding -= (lr * g_p).astype(np.float32)

            # Item positive gradients
            g_q_pos = -gamma * self.user_embedding + 1e-4 * q_pos
            g_b_pos = -gamma + 1e-4 * b_pos

            # Item negative gradients
            g_q_neg = gamma * self.user_embedding + 1e-4 * q_neg
            g_b_neg = gamma + 1e-4 * b_neg

            # DP L2 Clipping on Item Gradients
            norm_pos = float(np.sqrt(np.sum(g_q_pos ** 2) + g_b_pos ** 2))
            scale_pos = min(1.0, clip_c / (norm_pos + 1e-12))
            g_q_pos_clip = g_q_pos * scale_pos
            g_b_pos_clip = g_b_pos * scale_pos

            norm_neg = float(np.sqrt(np.sum(g_q_neg ** 2) + g_b_neg ** 2))
            scale_neg = min(1.0, clip_c / (norm_neg + 1e-12))
            g_q_neg_clip = g_q_neg * scale_neg
            g_b_neg_clip = g_b_neg * scale_neg

            # DP Gaussian Noise Injection
            if sigma_dp > 0.0:
                noise_q_pos = rng.normal(0.0, sigma_dp, size=g_q_pos_clip.shape).astype(np.float32)
                noise_b_pos = float(rng.normal(0.0, sigma_dp))
                noise_q_neg = rng.normal(0.0, sigma_dp, size=g_q_neg_clip.shape).astype(np.float32)
                noise_b_neg = float(rng.normal(0.0, sigma_dp))
                g_q_pos_final = g_q_pos_clip + noise_q_pos
                g_b_pos_final = g_b_pos_clip + noise_b_pos
                g_q_neg_final = g_q_neg_clip + noise_q_neg
                g_b_neg_final = g_b_neg_clip + noise_b_neg
            else:
                g_q_pos_final = g_q_pos_clip
                g_b_pos_final = g_b_pos_clip
                g_q_neg_final = g_q_neg_clip
                g_b_neg_final = g_b_neg_clip

            # Deltas for server
            delta_q_pos = -lr * g_q_pos_final
            delta_b_pos = -lr * g_b_pos_final
            delta_q_neg = -lr * g_q_neg_final
            delta_b_neg = -lr * g_b_neg_final

            # Accumulate deltas
            if pos_id not in touched_deltas_emb:
                touched_deltas_emb[pos_id] = delta_q_pos
                touched_deltas_bias[pos_id] = delta_b_pos
            else:
                touched_deltas_emb[pos_id] += delta_q_pos
                touched_deltas_bias[pos_id] += delta_b_pos

            if neg_id not in touched_deltas_emb:
                touched_deltas_emb[neg_id] = delta_q_neg
                touched_deltas_bias[neg_id] = delta_b_neg
            else:
                touched_deltas_emb[neg_id] += delta_q_neg
                touched_deltas_bias[neg_id] += delta_b_neg

        touched_idx = np.array(sorted(touched_deltas_emb.keys()), dtype=np.int64)
        delta_embs = np.array([touched_deltas_emb[idx] for idx in touched_idx], dtype=np.float32)
        delta_biases = np.array([touched_deltas_bias[idx] for idx in touched_idx], dtype=np.float32)
        mean_loss = float(np.mean(losses)) if losses else 0.0

        return touched_idx, delta_embs, delta_biases, mean_loss


class FederatedSimulation:
    """
    Master simulation engine managing rounds, client selection, DP parameter synchronization,
    and recommendation synthesis.
    """
    def __init__(self, dataset: SyntheticDataset, config: SimulationConfig):
        self.dataset = dataset
        self.config = config
        self.server = FederatedServer(dataset.num_items, config.embedding_dim, seed=config.seed)
        self.clients = [
            FederatedClient(
                user_id=u,
                train_items=np.where(dataset.train_matrix[u] > 0)[0],
                llm_profile=dataset.user_llm_profiles[u],
                embedding_dim=config.embedding_dim,
                seed=config.seed + u + 1
            )
            for u in range(dataset.num_users)
        ]
        self.current_round = 0
        self.history: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(config.seed)

        # Precompute target user popularity blueprints
        self.user_blueprints = compute_user_popularity_blueprint(
            dataset.train_matrix,
            dataset.head_idx,
            dataset.torso_idx,
            dataset.tail_idx
        )

    def step_round(self) -> Dict[str, Any]:
        """Advances simulation by 1 FL communication round."""
        self.current_round += 1
        active_candidates = [c.user_id for c in self.clients if len(c.train_items) > 0]
        if not active_candidates:
            active_candidates = list(range(len(self.clients)))

        n_sample = min(self.config.clients_per_round, len(active_candidates))
        sampled_ids = self.rng.choice(active_candidates, size=n_sample, replace=False).tolist()

        updates = []
        round_losses = []

        for cid in sampled_ids:
            client = self.clients[cid]
            t_idx, d_emb, d_b, loss = client.local_train_step(
                self.server.item_embeddings,
                self.server.item_biases,
                self.config,
                rng=self.rng
            )
            if len(t_idx) > 0:
                updates.append((t_idx, d_emb, d_b))
                round_losses.append(loss)

        # Server aggregation
        agg_stats = self.server.aggregate_updates(updates)
        avg_loss = float(np.mean(round_losses)) if round_losses else 0.0

        # Calculate DP sigma for telemetry
        if self.config.dp_enabled and self.config.dp_epsilon > 0:
            sigma_val = (self.config.dp_l2_clip_norm * np.sqrt(2.0 * np.log(1.25 / max(self.config.dp_delta, 1e-15)))) / max(self.config.dp_epsilon, 1e-5)
        else:
            sigma_val = 0.0

        round_record = {
            "round": self.current_round,
            "active_clients": sampled_ids,
            "loss": avg_loss,
            "items_touched_pct": agg_stats["items_touched_pct"],
            "dp_sigma": float(sigma_val),
            "dp_epsilon": float(self.config.dp_epsilon),
            "calibration_alpha": float(self.config.calibration_alpha),
            "llm_lambda": float(self.config.llm_lambda)
        }
        self.history.append(round_record)
        return round_record

    def get_recommendations(self, calibrated: bool = True) -> np.ndarray:
        """
        Synthesizes top-K recommendations for all users.
        If calibrated=False: raw uncalibrated base CF ranking.
        If calibrated=True: SUPER calibrated blueprint fusion ranking.
        """
        n_users = self.dataset.num_users
        n_items = self.dataset.num_items
        user_embs = np.array([c.user_embedding for c in self.clients], dtype=np.float32)

        # Base CF scores: (n_users, n_items)
        cf_scores = np.dot(user_embs, self.server.item_embeddings.T) + self.server.item_biases[None, :]

        if not calibrated:
            # Mask training items
            masked_scores = cf_scores.copy()
            masked_scores[self.dataset.train_matrix > 0] = -np.inf
            # Sort top K
            return np.argsort(-masked_scores, axis=1)[:, :self.config.top_k]

        # Calibrated SUPER Pipeline
        fused_scores = fuse_cf_and_llm_scores(
            cf_scores=cf_scores,
            user_profiles=self.dataset.user_llm_profiles,
            item_embeddings=self.dataset.item_llm_embeddings,
            head_idx=self.dataset.head_idx,
            torso_idx=self.dataset.torso_idx,
            tail_idx=self.dataset.tail_idx,
            llm_lambda=self.config.llm_lambda
        )

        recs = calibrated_blueprint_merge(
            fused_scores=fused_scores,
            user_blueprints=self.user_blueprints,
            train_matrix=self.dataset.train_matrix,
            head_idx=self.dataset.head_idx,
            torso_idx=self.dataset.torso_idx,
            tail_idx=self.dataset.tail_idx,
            top_k=self.config.top_k,
            calibration_alpha=self.config.calibration_alpha
        )
        return recs
