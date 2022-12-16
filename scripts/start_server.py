import os
from pathlib import Path
import sys

import argparse
import socket

sys.path.append(os.getcwd() + "/comgra")
from comgra import visualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--name', dest='name', default=None)
    parser.add_argument('--port', dest='port', default=None)
    parser.add_argument('--path', dest='path', default=None)
    args = parser.parse_args()
    if args.path is None:
        path = Path({
                        'fd-linux': Path(__file__).parent.parent.parent / 'comgra_data',
                        'ws76lx': "/data/users/fdietz/comgra_data"}.get(socket.gethostname(),
                                                                        "/data/users/fdietz/comgra_data")
                    ).absolute() / args.name
    else:
        path = Path(args.path).absolute() / args.name
    assert path.exists(), path
    vis = visualizer.Visualization(path=path)
    vis.run_server(args.port)
