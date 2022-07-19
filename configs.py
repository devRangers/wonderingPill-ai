import argparse

mn_parser = argparse.ArgumentParser(description="mobilenet")

mn_parser.add_argument("--batch_size", type=int, default=64)
mn_parser.add_argument("--gpu_id", type=int, default=-1)
mn_parser.add_argument("--n_classes", type=int)
mn_parser.add_argument("--verbose", type=int, default=1)
mn_parser.add_argument("--n_epochs", type=int, default=10)

mn_config = mn_parser.parse_args()