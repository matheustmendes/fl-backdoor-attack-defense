from pathlib import Path

from data import CLASS_NAMES, build_dataset, save_dataset

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "cache" / "celeba_defense_eval.pt"

NUM_SHARDS = 26
START_SHARD = 4
MAX_PER_CLASS = 1600


def main():
    dataset = build_dataset(
        num_shards=NUM_SHARDS,
        max_per_class=MAX_PER_CLASS,
        start_shard=START_SHARD,
    )
    save_dataset(dataset, CACHE_PATH)

    print(
        f"Saved {len(dataset)} images to {CACHE_PATH} "
        f"(shards {START_SHARD}-{START_SHARD + NUM_SHARDS - 1})"
    )

    counts = [0, 0, 0, 0]
    for label in dataset.labels.tolist():
        counts[label] += 1

    for name, count in zip(CLASS_NAMES, counts):
        print(f"  {name:6s}: {count}")

    import math

    print(
        f"\nbinomial std at p=0.53: +-{100 * math.sqrt(0.53 * 0.47 / len(dataset)):.2f}pp "
        f"(was +-3.35pp at n=222)"
    )


if __name__ == "__main__":
    main()
