from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "huggan/CelebA-faces-with-attributes"
IMAGE_SIZE = 64

CLASS_NAMES = ["black", "brown", "blond", "gray"]
ATTRIBUTE_COLUMNS = ["Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair"]


@dataclass(frozen=True)
class HairColorDataset:
    images: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        return self.images.shape[0]


def _is_valid_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"PAR1":
                return False

            handle.seek(-4, 2)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def download_shards(num_shards: int, start_shard: int = 0) -> list[Path]:
    paths = []
    skipped = []

    for shard_index in range(start_shard, start_shard + num_shards):
        filename = f"data/train-{shard_index:05d}-of-00132.parquet"
        path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))

        if _is_valid_parquet(path):
            paths.append(path)
        else:
            skipped.append(shard_index)

    if skipped:
        print(f"  warning: skipped {len(skipped)} corrupt upstream shard(s): {skipped}")

    if not paths:
        raise RuntimeError(
            f"Every shard in [{start_shard}, {start_shard + num_shards}) is corrupt."
        )

    return paths


def _decode_and_resize(jpeg_bytes: bytes) -> torch.Tensor:
    image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return TF.to_tensor(image)


def build_dataset(
    num_shards: int = 4,
    max_per_class: int | None = None,
    start_shard: int = 0,
) -> HairColorDataset:
    paths = download_shards(num_shards, start_shard)

    columns = ["image", *ATTRIBUTE_COLUMNS]
    tables = [pq.read_table(path, columns=columns) for path in paths]

    images: list[torch.Tensor] = []
    labels: list[int] = []
    per_class_count = [0, 0, 0, 0]

    for table in tables:
        rows = table.to_pylist()

        for row in rows:
            flags = [row[col] == 1 for col in ATTRIBUTE_COLUMNS]

            if sum(flags) != 1:
                continue

            label = flags.index(True)

            if max_per_class is not None and per_class_count[label] >= max_per_class:
                continue

            images.append(_decode_and_resize(row["image"]["bytes"]))
            labels.append(label)
            per_class_count[label] += 1

    return HairColorDataset(
        images=torch.stack(images),
        labels=torch.tensor(labels, dtype=torch.int64),
    )


def save_dataset(dataset: HairColorDataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": dataset.images, "labels": dataset.labels}, path)


def train_eval_split(
    dataset: HairColorDataset,
    eval_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[HairColorDataset, HairColorDataset]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator)

    num_eval = int(round(len(dataset) * eval_fraction))
    eval_indices = permutation[:num_eval]
    train_indices = permutation[num_eval:]

    train = HairColorDataset(
        images=dataset.images[train_indices],
        labels=dataset.labels[train_indices],
    )
    eval_ = HairColorDataset(
        images=dataset.images[eval_indices],
        labels=dataset.labels[eval_indices],
    )
    return train, eval_


def load_dataset(path: Path) -> HairColorDataset:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return HairColorDataset(images=payload["images"], labels=payload["labels"])
