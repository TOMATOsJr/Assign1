import argparse
import random
from pathlib import Path


def count_lines(path: Path) -> int:
	count = 0
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if line.strip():
				count += 1
	return count


def split_dataset(
	input_path: Path,
	train_path: Path,
	val_path: Path,
	test_path: Path,
	seed: int,
) -> None:
	total = count_lines(input_path)
	if total == 0:
		raise ValueError("Input file has no non-empty lines.")

	train_target = int(round(total * 0.60))
	val_target = int(round(total * 0.20))
	test_target = total - train_target - val_target

	remaining_total = total
	remaining_train = train_target
	remaining_val = val_target
	remaining_test = test_target

	rng = random.Random(seed)

	with (
		input_path.open("r", encoding="utf-8") as input_handle,
		train_path.open("w", encoding="utf-8") as train_handle,
		val_path.open("w", encoding="utf-8") as val_handle,
		test_path.open("w", encoding="utf-8") as test_handle,
	):
		processed = 0
		for line in input_handle:
			if not line.strip():
				continue
			processed += 1

			roll = rng.randint(1, remaining_total)
			if roll <= remaining_train:
				train_handle.write(line)
				remaining_train -= 1
			elif roll <= remaining_train + remaining_val:
				val_handle.write(line)
				remaining_val -= 1
			else:
				test_handle.write(line)
				remaining_test -= 1

			remaining_total -= 1

			if processed % 100000 == 0:
				print(f"Processed {processed} lines...")

	print("Split complete.")
	print(f"Total lines: {total}")
	print(f"Train: {train_target} -> {train_path}")
	print(f"Val: {val_target} -> {val_path}")
	print(f"Test: {test_target} -> {test_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Split JSONL into train/val/test")
	parser.add_argument(
		"--input",
		default="cc100_en_cleaned_final.jsonl",
		help="Input JSONL file",
	)
	parser.add_argument("--train", default="cc100_en_train.jsonl", help="Train output")
	parser.add_argument("--val", default="cc100_en_val.jsonl", help="Val output")
	parser.add_argument("--test", default="cc100_en_test.jsonl", help="Test output")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")

	args = parser.parse_args()

	split_dataset(
		input_path=Path(args.input),
		train_path=Path(args.train),
		val_path=Path(args.val),
		test_path=Path(args.test),
		seed=args.seed,
	)


if __name__ == "__main__":
	main()
