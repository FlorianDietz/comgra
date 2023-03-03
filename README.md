# comgra


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
