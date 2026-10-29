from pathlib import Path

from data import build_dataset, save_dataset

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "cache" / "celeba_subset.pt"


def main():
    dataset = build_dataset(num_shards=4, max_per_class=280)
    save_dataset(dataset, CACHE_PATH)

    print(f"Saved {len(dataset)} images to {CACHE_PATH}")

    counts = [0, 0, 0, 0]
    for label in dataset.labels.tolist():
        counts[label] += 1

    for name, count in zip(["black", "brown", "blond", "gray"], counts):
        print(f"  {name:6s}: {count}")


if __name__ == "__main__":
    main()
