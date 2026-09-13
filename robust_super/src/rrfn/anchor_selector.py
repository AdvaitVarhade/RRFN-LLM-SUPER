import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

class AnchorSelector:
    """
    Selects anchor points (instances where the model is highly confident in class k)
    to estimate the noise transition matrix T.
    """
    def __init__(
        self,
        anchor_percentile: float = 0.90,
        min_anchors_per_class: int = 20,
        num_classes: int = 5
    ):
        self.anchor_percentile = anchor_percentile
        self.min_anchors_per_class = min_anchors_per_class
        self.num_classes = num_classes

    def select_anchors(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = "cpu"
    ) -> Dict[int, List[Tuple[int, int, int, np.ndarray]]]:
        """
        Extracts high-confidence anchor points for each rating class (0-indexed).
        Returns: Dict[class_k, List of (user_id, item_id, observed_rating, probs_array)]
        """
        model.eval()
        anchor_sets: Dict[int, List[Tuple[int, int, int, np.ndarray]]] = {k: [] for k in range(self.num_classes)}
        all_samples: Dict[int, List[Tuple[int, int, int, np.ndarray, float]]] = {k: [] for k in range(self.num_classes)}

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) >= 3:
                    u_batch, i_batch, y_batch = batch[0], batch[1], batch[2]
                else:
                    continue

                u_b = u_batch.to(device)
                i_b = i_batch.to(device)
                probs = model(u_b, i_b)  # [B, 5]

                max_probs, pred_classes = torch.max(probs, dim=-1)

                probs_np = probs.cpu().numpy()
                max_probs_np = max_probs.cpu().numpy()
                pred_classes_np = pred_classes.cpu().numpy()
                u_np = u_batch.cpu().numpy()
                i_np = i_batch.cpu().numpy()
                y_np = y_batch.cpu().numpy()

                for idx in range(len(u_np)):
                    cls = int(pred_classes_np[idx])
                    conf = float(max_probs_np[idx])
                    sample = (int(u_np[idx]), int(i_np[idx]), int(y_np[idx]), probs_np[idx])
                    all_samples[cls].append((sample[0], sample[1], sample[2], sample[3], conf))

                    if conf >= self.anchor_percentile:
                        anchor_sets[cls].append(sample)

        # Ensure minimum anchors per class by falling back to top confident samples if needed
        for k in range(self.num_classes):
            if len(anchor_sets[k]) < self.min_anchors_per_class:
                # Sort available samples by confidence descending
                sorted_by_conf = sorted(all_samples[k], key=lambda x: -x[4])
                needed = self.min_anchors_per_class - len(anchor_sets[k])
                for s in sorted_by_conf[:needed]:
                    anchor_sets[k].append((s[0], s[1], s[2], s[3]))

        return anchor_sets
