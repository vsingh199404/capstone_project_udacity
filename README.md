# capstone_project_udacity
This is a kaggle competition project APTOS 2019 Blindness Detection


## Dataset download
Please visit this link to download data https://www.kaggle.com/c/aptos2019-blindness-detection/data

## Libraries and installaions require

### System requirement
- x64 bit operating system.
- Python 3.* version.
- GPU(preffered but not necessary)

This project requres the following libraries to be installed.

**OpenCV**

OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library

To install run cmd and copy past this command

```pip install opencv-python```


**Tesnorflow** and **Keras**

Follow these steps to install keras. It is recomended go trough official website for installaion steps.

1. Update conda and all the other packages.
~~~

# update conda in your default environment 
$ conda upgrade conda
$ conda upgrade --all
# create a new environment with conda
$ conda create -n [my-env-name]
$ conda cerate -n [my-env-name] python=[python-version]
# activate the environment you created
$ source activate [my-env-name]
# take a look at the environment you created
$ conda info
$ conda list
# install a package with conda and verify it's installed
$ conda install numpy
$ conda list
# take a look at the list of environments you currently have
$ conda info -e
# remove an environment
$ conda env remove --name [my-env-name]

~~~


2. Install pip
```

# install pip in the virtual environment
$ conda install pip

```

3. Install TensorFlow
```

# install Tensorflow CPU version
$ pip install --upgrade tensorflow # for python 2.7
$ pip3 install --upgrade tensorflow # for python 3.*

```


4. Install Keras
```

# install Keras (Note: please install TensorFlow first)
$ pip install Keras

```

## Directory structure

Make a input folder and uncompress the downloaded data datafiles in it. Also change the path to the input folder according to your system.
