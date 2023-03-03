import os
from pathlib import Path
import sys

import argparse
import socket

from comgra import recorder



def run_demonstration(comgra_data_root_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--name', dest='name', default=None)
    parser.add_argument('--port', dest='port', default=None)
    parser.add_argument('--path', dest='path', default=None)
    args = parser.parse_args()
    if args.path is None:
        path = (Path(__file__).parent.parent / 'testing_data').absolute() / args.name
        path.mkdir(exist_ok=True, parents=True)
    else:
        path = Path(args.path).absolute() / args.name
    assert path.exists(), path
    run_demonstration(path)
