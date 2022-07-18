import argparse
import sys

from ..config.defaults import get_cfg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for NextLocationPrediction training and testing'
    )

    parser.add_argument('--cfg', dest="cfg_file", help="Path to the config file", type=str)

    parser.add_argument("opts", default=None, help = "See config/defaults.py for all options", nargs = argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    
    return cfg