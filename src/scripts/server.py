from pathlib import Path

import argparse
import multiprocessing
import time
from typing import Optional

from comgra import visualizer

restart_signal_queue = multiprocessing.Queue()

def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--path', dest='path', default=None)
    parser.add_argument('--port', dest='port', default=8055)
    parser.add_argument('--visualization-file', dest='external_visualization_file', default=None)
    parser.add_argument('--use-path-for-test-run', dest='use_path_for_test_run', default=False, action='store_true')
    parser.add_argument(
        '--debug-mode', dest='debug_mode', default=False, action='store_true',
        help="Run Dash in debug mode. Note that this also enables automatic reloading on code change (use_reloader),"
             "which interferes with multiprocessing and breaks the button for reloading the server."
    )
    args = parser.parse_args()
    assert (args.path is None) is args.use_path_for_test_run, \
        "Either provide --path or set --use-path-for-test-run."
    if args.use_path_for_test_run:
        args.path = (Path(__file__).parent.parent.parent / 'testing_data' / 'testcase_for_demonstration').absolute()
        args.external_visualization_file = (Path(__file__).parent / 'example_custom_visualization.py').absolute()
    full_path = Path(args.path).absolute()
    assert full_path.exists(), full_path
    args.full_path = full_path

    # Either run in a loop that allows the user to manually restart the server,
    # or use debug_mode and Dash's built-in use_reloader.
    the_server_process: Optional[multiprocessing.Process] = None

    def start_or_restart_server():
        nonlocal the_server_process
        if the_server_process is not None:
            the_server_process.terminate()
            the_server_process.join()
        the_server_process = multiprocessing.Process(target=run_visualization, args=(args,))
        the_server_process.start()
    if args.debug_mode:
        run_visualization(args)
    else:
        start_or_restart_server()
        # Infinite loop to keep the main thread alive,
        # while listening for a signal to restart the server
        while True:
            msg = restart_signal_queue.get()
            if msg == 'restart':
                start_or_restart_server()

def run_visualization(args):
    vis = visualizer.Visualization(
        path=args.full_path,
        debug_mode=args.debug_mode,
        external_visualization_file=args.external_visualization_file,
        restart_signal_queue=restart_signal_queue,
    )
    vis.run_server(args.port, use_reloader=args.debug_mode)


if __name__ == '__main__':
    main()
