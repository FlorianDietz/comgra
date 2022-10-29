import os
from pathlib import Path
import sys

import argparse

sys.path.append(os.getcwd() + "/comgra")
from comgra import visualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--name', dest='name', default=None)
    args = parser.parse_args()
    path = Path("/home/remote/PycharmProjects/AI22/comgra_data") / args.name
    assert path.exists(), path
    vis = visualizer.Visualization(path=path)
    vis.run_server()
