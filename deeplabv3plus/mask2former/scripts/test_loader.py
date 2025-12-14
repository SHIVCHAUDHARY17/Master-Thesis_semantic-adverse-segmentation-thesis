import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from datasets.cityscapes import Cityscapes

def custom_collate(batch):
    return {
        "image": [item["image"] for item in batch],
        "mask": [item["mask"] for item in batch],
        "weather": [item["weather"] for item in batch],
        "time": [item["time"] for item in batch]
    }

dataset = Cityscapes(
    root="/home/zboxubuntu22/datasets/cityscapes",
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=custom_collate
)

batch = next(iter(loader))
print("Batch keys:", batch.keys())
print("Image type:", type(batch["image"]))
print("First image type:", type(batch["image"][0]))
