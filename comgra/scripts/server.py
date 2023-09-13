import os
from pathlib import Path
import sys

import argparse
import socket

from comgra import visualizer

def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--path', dest='path', default=None)
    parser.add_argument('--port', dest='port', default=8055)
    parser.add_argument('--use-path-for-test-run', dest='use_path_for_test_run', default=False, action='store_true')
    args = parser.parse_args()
    assert (args.path is None) is args.use_path_for_test_run, \
        "Either provide --path or set --use-path-for-test-run."
    if args.use_path_for_test_run:
        args.path = (Path(__file__).parent.parent.parent / 'testing_data' / 'publication_test').absolute()
    path = Path(args.path).absolute()
    assert path.exists(), path
    vis = visualizer.Visualization(path=path)
    vis.run_server(args.port)

if __name__ == '__main__':
    main()
