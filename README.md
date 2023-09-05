# comgra

## Debugging Neural Networks more easily

Comgra stands for "computation graph analysis".

It is a library for use with pytorch that makes it easier to inspect the internals of your neural networks.

Debugging neural architectures can be difficult, as the computation graphs are often very large and complex. This tool allows you to visualize the computation graph and inspect the values of individual tensors at different points in time.

You can give names to tensors in your code, and they will be recorded and can be visualized graphically.

This allows you to compare tensors as training proceeds.

Tensors can be analyzed with different degrees of granularity depending on your needs. Inspect only KPIs across an entire batch, or drill down into the values of individual neurons on a single sample.

You can also visualize which tensors are dependent on which other tensors, and track their gradients.

This tool is especially helpful if you are developing new architectures and the architectures show unexpected behavior, as this visualization can help you understand what is going on much faster than the typical graph visualizations can do on their own. It is often useful to look at performance graphs of tensorboard or similar tools to identify which training steps have unexpected behavior, and then switch to comgra to inspect those steps in detail.

See this screenshot for what the visualization looks like. Each rectangle is called a Node. They are clickable and each represents a tensor. The selection and sliders below specify which version of that tensor you want to inspect. The Nodes are causally connected, where the Nodes to the left appear earlier during computation and those to the right appear later. When you select a Node, all Nodes that are directly connected to it in the computation graph become highlighted with a border color (I have found that this is easier to read than actually drawing the connections as arrows once the number of Nodes grows large enough).

![Example screenshot of comgra](comgra_screenshot.png?raw=true "Example screenshot of comgra")


## Installation

Run this to install:

`pip install comgra`

## Testing

Use the files in the 'testing' folder to run a basic ML model and visualize it.

`testing/run.py` will create a folder to store recordings.
`testing/server.py` will open that folder and visualize it.

In both cases, the --path argument can be used to change the default path to store data, which is in the directory where comgra is installed.

## Usage

`testing/run.py` contains documentation for how to use comgra in your own code. If anything is unclear, I will be happy to answer any questions at floriandietz44@gmail.com 

The script takes two parameters:
--path should be a folder where all your comgra results are stored.
--name will be the name for this particular recording of comgra. A folder with this name will be created in --path, and it will overwrite previous results with the same name.

You can visualize the results using the script in `testing/server.py`.

This takes three parameters:
--path should be a folder where all your comgra results are stored.
--name is the name for the recording you want to display.
--port is the port of localhost where the results will be displayed.


## Future Development


A goal for the future development of this tool is the automated detection of anomalies in computational graphs.

It should be possible to define anomalies like "Tensor X has a greater absolute value than 1" or the like, and then have the program automatically calculate likely dependencies such as this:

The anomaly "abnormally high loss" has 87% correlation with the anomaly "Tensor 5 is close to zero".

This would save a lot of time with debugging, by automatically generating a list of possible reasons for unexpected behavior.

Anomalies could be defined in many different ways, and standard tools for causality analysis already exist. If you are interested in a feature like this and/or want to help, please let me know at floriandietz44@gmail.com

