import argparse

mn_parser = argparse.ArgumentParser(description="mobilenet")

mn_parser.add_argument("--batch_size", type=int, default=64)

mn_config = mn_parser.parse_args()