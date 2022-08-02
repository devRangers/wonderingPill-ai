import argparse

p = argparse.ArgumentParser(description="mobilenet")

p.add_argument("--gpu_id", type=int, default=-1)

p.add_argument("--shape_classifier", type=lambda s: s.lower() in ["true", "1"])

p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--verbose", type=int, default=1)
p.add_argument("--n_epochs", type=int, default=10)

p.add_argument("--n_classes", type=int, default=9)


mn_config = p.parse_args()
