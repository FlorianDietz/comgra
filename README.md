# Comgra: Computation Graph Analysis

<p align="center">
<img src="src/assets/brandcrowd_logos/FullLogo.png" title="ComgraLogo" height="300" width="300"/>
</p>

Comgra helps you analyze and debug neural networks in pytorch.  
It records your network internals, visualizes the computation graph, and provides a GUI that makes it fast and easy to investigate any part of your network from a variety of viewpoints.  
Move along the computation graph, check for outliers, investigate both individual data points and summary statistics, compare gradients, automatically record special cases, and more.

Comgra is complementary to tensorboard:  
Use Tensorboard to get an overview of summary statistics, so you understand what is happening at a high level.  
Use Comgra to deep dive into your neural network: Comgra records everything that could be relevant to you at a low overhead, and provides a flexible GUI that allows you to inspect your network's behavior from many different angles.

Suitable both for novices and for professional neural architecture designers: Create a simple visualization of your network to understand what is happening under the hood, or perform advanced analyses and trace anomalies through the computation graph.

TODO
![main_overview_version_1.png](src%2Fassets%2Fscreenshots_for_tutorial%2Fmain_overview_version_1.png)
TODO add notes to screenshot ; what is displayed here?
this shows one iteration of one training step ; different iterations have different layouts
This graph is a subgraph of the computation graph, and it is much easier to understand because it is smaller and skips all of the distracting details.
This cutting away of details also makes it easier to compare different variants of architectures: Their computation graphs may look different, but the simplified dependency graphs are the same.
While the dependency graph is generated automatically, it can also be customized to be more readable and easier to navigate if necessary.
Each rectangle in the dependency graph is a node that represents a named tensor that can be selected for inspection. The colors indicate the roles of the tensor in the network, such as input, intermediate, parameter, etc.
Look at the run code for detailed examples. The following shows how that works

## In this README

TODO
- [Quick start](#quick-start)
- [Installation](#installation)
  - [Installing with PIP](#installing-with-pip)
  - [Installing with Conda](#installing-with-conda)
  - [Installing from source](#installing-from-source)
  - [Checking your installation](#checking-your-installation)
  - [Docker image](#docker-image)
- [FAQ](#faq)
- [Team](#team)
- [License](#license)

## Installation

Install with pip:

```bash
pip install comgra
```

## Usage

To use comgra, modify your python code with the following commands in the appropriate places. This may look daunting, but most of it really just tells comgra what you are currently doing so that it knows how to associate the tensors you register. The file `src/scripts/run.py` contains a documented example that you can copy and will be explained in detail below.

```python
import comgra
from comgra.recorder import ComgraRecorder
# Define a recorder
comgra.my_recorder = ComgraRecorder(...)
# Track your network parameters
comgra.my_recorder.track_module(...)
# Optionally, add some notes for debugging
comgra.my_recorder.add_note(...)
# Call this whenever you start a new training step you want to record:
comgra.my_recorder.start_next_recording(...)
# Call this whenever you start the forward pass of an iteration. In multi-iteration experiments, call it once per iteration:
comgra.my_recorder.start_forward_pass(...)
# Register any tensors you care about:
comgra.my_recorder.register_tensor(...)
# Call these whenever you apply losses and propagate gradients:
comgra.my_recorder.start_backward_pass()
comgra.my_recorder.record_current_gradients(...)
# Call this whenever you end an iteration:
comgra.my_recorder.finish_iteration()
# Call this whenever you end a training step:
comgra.my_recorder.finish_batch()
```

When your code runs, comgra will store data in the folder you specified with `ComgraRecorder(comgra_root_path="/my/path/for/storing/data")`.  
In the process, it will automatically organize everything, extract statistics, and build the dependency graph.

To start the GUI and visualize your results, run
```bash
comgra --path "/my/path/for/storing/data"
```


## Tutorial - Debugging an example network

The file `src/scripts/run.py` trains a neural network on an example task. This network contains a subtle bug, and in this tutorial we will show how you can use comgra to find that bug.

For convenience, you can run the file from the commandline using
```bash
comgra-test-run
```

and you can start the GUI on the data it generates by calling
```bash
comgra --use-path-for-test-run
```

### The task and the architecture

We use a synthetic task that is designed to test a neural network's ability to generalize to longer sequences while being very simple and human-interpretable.  
The input is a sequence of N tuples of 5 numbers between 0.0 and 1.0. The network should treat these as 5 separate sequences. Its objective is to determine which of these 5 sequences has the largest sum.

Our architecture is a simple recurrent neural network that is composed of some submodules. It's nothing fancy, but illustrates how comgra can be integrated into an architecture.  
We run two variants of the architecture. The original variant contains a bug, which we will discover later in this section of the Readme. For convenience we run both trials in one script, but in a real use case the second variant would have been implemented and run later, after finding the bug. In the GUI, you can switch between the two variants with the 'Trial' selector.

### Initial exploration

As a first step, let's look at network summary information and the notes created by the script. To do so, select "Network" and "Notes" respectively at the main radio button at the top left of the screen.

<p align="center">
  <img src="src/assets/screenshots_for_tutorial/network_info.png" width="40%"/>
  <img src="src/assets/screenshots_for_tutorial/notes_info.png" width="40%"/>
</p>

<details>
  <summary>Walking through the computation graph</summary>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/00_start.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/01_selectors_set.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/02.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/03.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/04.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/05.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/06.png" width="500"/>
  <img src="src/assets/screenshots_for_tutorial/slideshow_nodes/07.png" width="500"/>
</details>


### Finding the bug




## Testing

Installing comgra through pip also installs two scripts:

`comgra-test-run` to verify it was installed correctly and `comgra --use-path-for-test-run` for starting the server.

## Usage

`comgra-test-run` calls the file `src/scripts/run.py`, which contains documentation for how to use comgra in your own code. If anything is unclear, I will be happy to answer any questions at floriandietz44@gmail.com 

The script takes two optional parameters:
--path should be a folder where all your comgra results are stored.
--group will be the name for this particular recording of comgra. A folder with this name will be created in --path, and it will overwrite previous results with the same name.

For convenience, if you don't specify these parameters, the results will be stored in the library's folder instead.

You can visualize the results using the script in `src/scripts/server.py`, which you can call with `comgra --use-path-for-test-run`.

This takes three parameters:
--path should be a folder where all your comgra results are stored. If you called `comgra-test-run` with its --path and --group parameters, this path variable should include the --group as well.
--port is the port of localhost where the results will be displayed.
--use-path-for-test-run is a flag to ignore the --path and use a hardcoded path instead, for convenient testing.


## Future Development


A goal for the future development of this tool is the automated detection of anomalies in computational graphs.

It should be possible to define anomalies like "Tensor X has a greater absolute value than 1" or the like, and then have the program automatically calculate likely dependencies such as this:

The anomaly "abnormally high loss" has 87% correlation with the anomaly "Tensor 5 is close to zero".

This would save a lot of time with debugging, by automatically generating a list of possible reasons for unexpected behavior.

Anomalies could be defined in many different ways, and standard tools for causality analysis already exist. If you are interested in a feature like this and/or want to help, please let me know at floriandietz44@gmail.com

