# comgra

## What is this?

Comgra stands for "computation graph analysis".

It is a library for use with pytorch that makes it easier to inspect the intermediate results of your experiments.

You can give names to tensors in your code, and they will be recorded and can be visualized graphically.

This allows you to compare tensors as training proceeds.

Tensors can be analyzed with different degrees of granularity depending on your needs. Inspect only KPI across an entire batch, or drill down into the values of individual neurons on a particular test set.

You can also visualize which tensors are dependent on which other tensors, and track their gradients.

This tool is helpful for debugging the behavior of neural networks. It is especially helpful if you are developing new architectures and the architectures show unexpected behavior, as this visualization can help you understand what is going on much faster than the typical graph visualizations can do on their own. It is often useful to look at performance graphs of tensorboard or similar tools to identify which training steps have unexpected behavior, and then switch to comgra to inspect those steps in detail.

See this screenshot for what the visualization looks like. Each rectangle is clickable and represents a tensor. The selection and sliders below specify which version of that tensor you want to inspect.
T
![Example screenshot of comgra](comgra_screenshot.png?raw=true "Example screenshot of comgra")


## Development


A goal for the future development of this tool is the automated detection of anomalies in computational graphs.

It should be possible to define anomalies like "Tensor X has a greater absolute value than 1" or the like, and then have the program automatically calculate likely dependencies such as this:

The anomaly "abnormally high loss" has 87% correlation with the anomaly "Tensor 5 is close to zero".

This would save a lot of time with debugging, by automatically generating a list of possible reasons for unexpected behavior.

Anomalies could be defined in many different ways, and standard tools for causality analysis already exist. If you are interested in a feature like this, please let me know at floriandietz44@gmail.com as I will only extend this tool if I see enough of a need for it.


## Installation instructions

Run this to install:

`pip install comgra`

## Testing instructions

Use the files in the 'testing' folder to run a basic ML model and visualize it.

`testing/run.py` will create a folder to store recordings.
`testing/server.py` will open that folder and visualize it.

In both cases, the --path argument can be used to change the default path to store data, which is in the directory where comgra is installed.

## How to use this.

`testing/run.py` contains documentation for how to use comgra in your own code. This does not cover all special cases yet because this library is under development. If you have questions, just write me an email at floriandietz44@gmail.com and I will be happy to answer it and update the documentation.

This takes three parameters:
--path should be a folder where all your comgra results are stored.
--name will be the name for this particular recording of comgra. A folder with this name will be created in --path, and it will overwrite previous results with the same name.

You can visualize the results using the script in `testing/server.py`.

This takes three parameters:
--path should be a folder where all your comgra results are stored.
--name is the name for the recording you want to display.
--port is the port of localhost where the results will be displayed.
