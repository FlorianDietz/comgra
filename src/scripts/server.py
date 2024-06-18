from pathlib import Path

import argparse
import multiprocessing
import time
from typing import Optional

from comgra import visualizer

def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--path', dest='path', default=None)
    parser.add_argument('--port', dest='port', default=8055)
    parser.add_argument('--visualization-file', dest='external_visualization_file', default=None)
    parser.add_argument('--use-path-for-test-run', dest='use_path_for_test_run', default=False, action='store_true')
    parser.add_argument('--debug-mode', dest='debug_mode', default=False, action='store_true')
    args = parser.parse_args()
    assert (args.path is None) is args.use_path_for_test_run, \
        "Either provide --path or set --use-path-for-test-run."
    if args.use_path_for_test_run:
        args.path = (Path(__file__).parent.parent.parent / 'testing_data' / 'testcase_for_demonstration').absolute()
        args.external_visualization_file = (Path(__file__).parent / 'example_custom_visualization.py').absolute()
    path = Path(args.path).absolute()
    assert path.exists(), path
    the_visualization: Optional[visualizer.Visualization] = multiprocessing.shared_memory
    the_server_process: Optional[multiprocessing.Process] = None

    def run_visualization():
        nonlocal the_visualization
        the_visualization = visualizer.Visualization(path=path, debug_mode=args.debug_mode, external_visualization_file=args.external_visualization_file)
        the_visualization.run_server(args.port)

    def start_or_restart_server():
        nonlocal the_server_process
        if the_server_process is not None:
            print(1)
            print(2)
            the_server_process.terminate()
            the_server_process.join()
            print(3)
        the_server_process = multiprocessing.Process(target=run_visualization)
        print(4, the_server_process)
        the_server_process.start()

    start_or_restart_server()
    time.sleep(5)
    start_or_restart_server()
    # Infinite loop to keep the main thread alive,
    # while listening for a signal to restart the server
    # TODO enable restarts
    #  to do so, use multiprocessing.shared_memory
    # TODO make the button trigger a page refresh
    # TODO Add an error message when reloading all data while graphs changed
    while True:
        time.sleep(1)
        print(the_visualization)
        if the_visualization is not None and the_visualization.SIGNAL_TO_RESTART_SERVER:
            print("RESTARt")
            start_or_restart_server()
            the_visualization = None

if __name__ == '__main__':
    main()
