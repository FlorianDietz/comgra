# comgra


## Installation instructions

(Much of the below is not necessary after all, if you use a website instead of jupyter to display things.)
(Likewise some requirements are probably unnecessary)
(TODO: Clean these up once I am sure whether or not to use jupyter for visualization after all)

Install nodejs.

jupyter nbextension install --sys-prefix --symlink --py jupyter_dash
jupyter nbextension enable jupyter_dash
jupyter labextension link extensions/jupyterlab
jupyter serverextension enable jupyter_server_proxy

jupyter lab build