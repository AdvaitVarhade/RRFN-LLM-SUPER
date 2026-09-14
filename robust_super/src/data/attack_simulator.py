import copy
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from .preprocessor import DataSplit

class AttackSimulator:
    """
    Simulates various data poisoning, rating corruption, and review-bombing attacks on recommender datasets.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def inject_attack(
        self,
        split: DataSplit,
        attack_type: str,
        noise_rate: float,
        target_items: Optional[List[int]] = None,
        time_window_hours: int = 24
    ) -> DataSplit:
        """
        Injects the specified attack into a copy of the DataSplit training data.
        Returns a new corrupted DataSplit with updated ground_truth_labels.
        """
        attacked = copy.deepcopy(split)
        if attack_type == "none" or noise_rate <= 0.0:
            return attacked

        if attack_type == "random_flip":
            self._inject_random_flip(attacked, noise_rate)
        elif attack_type == "bandwagon":
            self._inject_bandwagon(attacked, noise_rate, target_items)
        elif attack_type == "nuke_bomb":
            self._inject_nuke_bomb(attacked, noise_rate, target_items, time_window_hours)
        elif attack_type == "agas":
            self._inject_agas(attacked, noise_rate, target_items, time_window_hours)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Recompute item counts on attacked train dict
        new_counts = {}
        for u, interactions in attacked.train_dict.items():
            for item, rating, ts in interactions:
                new_counts[item] = new_counts.get(item, 0) + 1
        attacked.item_counts = new_counts

        return attacked

    def _inject_random_flip(self, split: DataSplit, noise_rate: float):
        """Randomly flips ratings to an alternative value with probability noise_rate."""
        for u, interactions in split.train_dict.items():
            new_interactions = []
            for item, rating, ts in interactions:
                if np.random.rand() < noise_rate:
                    possible_ratings = [r for r in [1, 2, 3, 4, 5] if r != rating]
                    flipped_rating = int(np.random.choice(possible_ratings))
                    new_interactions.append((item, flipped_rating, ts))
                    split.ground_truth_labels[(u, item)] = 0  # corrupted
                else:
                    new_interactions.append((item, rating, ts))
            split.train_dict[u] = new_interactions

    def _inject_bandwagon(self, split: DataSplit, attack_rate: float, target_items: Optional[List[int]]):
        """
        Bandwagon Shilling Attack:
        Injects fake user profiles rating popular head items (filler) with 4-5★
        and target tail items with 5★.
        """
        # Identify top popular items (fillers) and lowest tail items (targets)
        sorted_items = sorted(split.item_counts.items(), key=lambda x: -x[1])
        all_item_ids = [i for i, _ in sorted_items]
        
        top_head_fillers = all_item_ids[:min(50, len(all_item_ids))]
        tail_candidates = all_item_ids[-min(100, len(all_item_ids)):]

        if not target_items:
            target_items = list(np.random.choice(tail_candidates, size=min(5, len(tail_candidates)), replace=False))

        num_fake_users = max(1, int(attack_rate * split.num_users))
        median_ts = int(np.median([ts for u_list in split.train_dict.values() for _, _, ts in u_list]))

        for fake_idx in range(num_fake_users):
            fake_user_id = split.num_users + fake_idx
            fake_interactions = []

            # 1. Filler items (15 to 30 items)
            num_fillers = np.random.randint(15, min(31, len(top_head_fillers)))
            chosen_fillers = np.random.choice(top_head_fillers, size=num_fillers, replace=False)
            for f_item in chosen_fillers:
                rating = int(np.random.choice([4, 5], p=[0.3, 0.7]))
                ts = median_ts + np.random.randint(-86400, 86400)
                fake_interactions.append((int(f_item), rating, ts))
                split.ground_truth_labels[(fake_user_id, int(f_item))] = 0
                split.weight_dict[(fake_user_id, int(f_item))] = 1.0

            # 2. Target items (Push: 5★)
            for t_item in target_items:
                ts = median_ts + np.random.randint(-86400, 86400)
                fake_interactions.append((int(t_item), 5, ts))
                split.ground_truth_labels[(fake_user_id, int(t_item))] = 0
                split.weight_dict[(fake_user_id, int(t_item))] = 1.0

            split.train_dict[fake_user_id] = fake_interactions
            split.user_mean_ratings[fake_user_id] = 4.8

        split.num_users += num_fake_users

    def _inject_nuke_bomb(self, split: DataSplit, attack_rate: float, target_items: Optional[List[int]], time_window_hours: int):
        """
        Nuke / Review Bombing Attack:
        Injects coordinated 1★ ratings on target popular head items within a tight temporal burst.
        """
        sorted_items = sorted(split.item_counts.items(), key=lambda x: -x[1])
        top_head_items = [i for i, _ in sorted_items[:min(20, len(sorted_items))]]

        if not target_items:
            target_items = list(np.random.choice(top_head_items, size=min(3, len(top_head_items)), replace=False))

        num_fake_users = max(1, int(attack_rate * split.num_users))
        burst_center_ts = int(np.median([ts for u_list in split.train_dict.values() for _, _, ts in u_list]))
        window_seconds = time_window_hours * 3600

        for fake_idx in range(num_fake_users):
            fake_user_id = split.num_users + fake_idx
            fake_interactions = []

            # 1. Random fillers (3 to 8 items)
            num_fillers = np.random.randint(3, 9)
            filler_pool = [i for i, _ in sorted_items[20:]]
            if filler_pool:
                chosen_fillers = np.random.choice(filler_pool, size=min(num_fillers, len(filler_pool)), replace=False)
                for f_item in chosen_fillers:
                    r = int(np.random.choice([3, 4, 5]))
                    ts = burst_center_ts + np.random.randint(-window_seconds // 2, window_seconds // 2)
                    fake_interactions.append((int(f_item), r, ts))
                    split.ground_truth_labels[(fake_user_id, int(f_item))] = 0
                    split.weight_dict[(fake_user_id, int(f_item))] = 1.0

            # 2. Target items (Nuke: 1★ clustered in time)
            for t_item in target_items:
                ts = burst_center_ts + np.random.randint(-window_seconds // 4, window_seconds // 4)
                fake_interactions.append((int(t_item), 1, ts))
                split.ground_truth_labels[(fake_user_id, int(t_item))] = 0
                split.weight_dict[(fake_user_id, int(t_item))] = 1.0

            split.train_dict[fake_user_id] = fake_interactions
            split.user_mean_ratings[fake_user_id] = 1.5

        split.num_users += num_fake_users

    def _inject_agas(self, split: DataSplit, attack_rate: float, target_items: Optional[List[int]], time_window_hours: int):
        """
        AGAS (Agentic Group Shilling Attack - ICDM 2026):
        Simulates coordinator-worker attack with two diverse worker groups and multi-round activation.
        """
        sorted_items = sorted(split.item_counts.items(), key=lambda x: -x[1])
        tail_candidates = [i for i, _ in sorted_items[-min(150, len(sorted_items)):]]

        if not target_items:
            target_items = list(np.random.choice(tail_candidates, size=min(3, len(tail_candidates)), replace=False))

        num_fake_users = max(2, int(attack_rate * split.num_users))
        half_users = num_fake_users // 2
        burst_ts = int(np.median([ts for u_list in split.train_dict.values() for _, _, ts in u_list]))

        # Group 1: Moderate fillers + 5★ target
        top_pool = [i for i, _ in sorted_items[:100]]
        g1_size = min(15, len(top_pool))
        for idx in range(half_users):
            fake_id = split.num_users + idx
            fake_list = []
            chosen_fillers = np.random.choice(top_pool, size=g1_size, replace=len(top_pool) < g1_size) if g1_size > 0 else []
            for f in chosen_fillers:
                r = int(np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
                ts = burst_ts + np.random.randint(-86400, 0)
                fake_list.append((int(f), r, ts))
                split.ground_truth_labels[(fake_id, int(f))] = 0
            for t in target_items:
                fake_list.append((int(t), 5, burst_ts))
                split.ground_truth_labels[(fake_id, int(t))] = 0
            split.train_dict[fake_id] = fake_list
            split.user_mean_ratings[fake_id] = 4.2

        # Group 2: Niche fillers + 5★ target (Round 2 adaptation)
        mid_pool = [i for i, _ in sorted_items[100:300]]
        if not mid_pool:
            mid_pool = [i for i, _ in sorted_items[max(0, len(sorted_items)//2):]] or [i for i, _ in sorted_items]
        g2_size = min(10, len(mid_pool))
        for idx in range(half_users, num_fake_users):
            fake_id = split.num_users + idx
            fake_list = []
            chosen_fillers = np.random.choice(mid_pool, size=g2_size, replace=len(mid_pool) < g2_size) if g2_size > 0 else []
            for f in chosen_fillers:
                r = int(np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3]))
                ts = burst_ts + np.random.randint(0, 86400)
                fake_list.append((int(f), r, ts))
                split.ground_truth_labels[(fake_id, int(f))] = 0
            for t in target_items:
                fake_list.append((int(t), 5, burst_ts + 43200))
                split.ground_truth_labels[(fake_id, int(t))] = 0
            split.train_dict[fake_id] = fake_list
            split.user_mean_ratings[fake_id] = 3.6

        split.num_users += num_fake_users
