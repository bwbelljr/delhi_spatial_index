# Delhi Public Services Index

## Description

This repository includes code to generate a spatial index of services for Delhi, based on the forthcoming paper "Towards an Urban Public Services Index" from Georgetown's [Urban Spatial Observatory](https://www.urbanspatialobservatory.org/). Although the data is not provided in this repository, the scripts and Jupyter notebooks can be used to generate an urban public services index in another city.

# Setup Notebook

* Install git on your machine.
    * Check by going to your terminal and typing `git version`
    * Instructions to setup here: https://git-scm.com
* Clone git repository: `git clone https://github.com/bwbelljr/delhi_spatial_index.git`
* Navigate to the repository folder
* Change branch: `git checkout bb_update`
* Install docker on your machine
    * Check if it is installed in your terminal by typing `docker version`
    * Instructions to setup docker here: https://docs.docker.com/engine
* In your terminal, type `docker image build -t jupyternotebook .`
* To launch the notebook, type: `docker container run -p 8888:8888 jupyternotebook`
* Find the link that starts with http://127.0.0.1:8888, copy to your browser, and start
* You may need to zip your data files, upload to Jupyter notebook and unzip.
