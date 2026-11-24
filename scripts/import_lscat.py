import argparse
import json

def main() -> None:
    from data.lscat_adapter import import_lscat_1d




    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="data/raw/ls-cat-official")
    parser.add_argument("--archive")
    parser.add_argument("--output-dir", default="data/processed/lscat_t4_1d")
    args = parser.parse_args()
    summary = import_lscat_1d(
        args.repository,
        args.output_dir,
        archive=args.archive,
    )
    print(f"WE got {args.output_dir}/runtimes.csv {args.output_dir}/split_groups.json")
    print(summary)


if __name__ == "__main__":
    main()
