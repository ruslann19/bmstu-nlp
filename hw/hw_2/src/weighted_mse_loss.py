import collections
import torch
from torch import nn
from typing import Dict, Callable


class WeightedMSELoss(nn.Module):
    def __init__(
        self,
        class_counts: Dict[int, int],
    ) -> None:
        super().__init__()
        class_counts = collections.OrderedDict(sorted(class_counts.items()))

        self.class_counts = class_counts
        total = sum(class_counts.values())

        self.weights = {}
        for class_id, count in class_counts.items():
            self.weights[class_id] = total / count

        # print("Веса:", self.weights)

    def forward(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
    ) -> torch.tensor:
        squared_errors = (predictions - targets) ** 2

        batch_weights = torch.ones_like(targets)
        for cls, weight in self.weights.items():
            mask = targets == cls
            batch_weights[mask] = weight

        return (squared_errors * batch_weights).mean()


# # Использование
# class_counts = train_df["labels"].astype(int).value_counts().to_dict()

# loss_fn = WeightedMSELoss(class_counts)


def create_loss_fn(
    loss_type: str,
    class_counts: Dict[int, int],
) -> Callable:
    match loss_type:
        case "MSELoss":
            return nn.MSELoss()
        case "WeightedMSELoss":
            return WeightedMSELoss(class_counts)
