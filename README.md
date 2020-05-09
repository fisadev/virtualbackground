# ViBa: virtual backgrounds

Fake webcam that replaces your background with a custom image of your choice.

This is a **work in progress**, although it does work if you have all the right dependencies installed.

No nice cli, but for now you can customize devices and other params in the last few lines of `viba.py`.

# Installation

### 1. Fake webcam device

You will need to do some weird stuff first: adding a kernel module to be able to have "fake" webcams.
This is done using the `v4l2loopback-dkms` package. On Debian or Ubuntu, you can install and configure it like this:

    # install the virtual webcams module
    sudo apt-get install v4l2loopback-dkms
    # create a config so the module is loaded and a cam is created on boot
    echo options v4l2loopback devices=1 video_nr=20 \
    card_label="viba_cam" exclusive_caps=1 | sudo tee -a \
    /etc/modprobe.d/viba_cam.conf
    echo v4l2loopback | sudo tee -a /etc/modules-load.d/viba_cam.conf
    # enable the module for the first time
    sudo modprobe -r v4l2loopback
    sudo modprobe v4l2loopback

This means that from now on, you will have a `/dev/video20` device that simulates being a webcam.

### 2. Python libs

Then you need to install the dependencies in the `requirements.txt` file, as usual with python. 
A virtualenv is recommended:

    python3.7 -m venv my_viba_venv
    source my_viba_venv/bin/activate
    pip install pip --upgrade
    pip install -r requirements.txt


# Usage

Just run the `viba.py` script, with your virtualenv activated:

    source my_viba_venv/bin/activate
    python viba.py

The script allows for a number of params to customize how the virtual background works:

**TODO**

# Usage as a lib

You can also use it from your own programs. Just take a look at the `main` function in `viba.py`, that's a perfect example and very simple.

# SPEEEED: using GPU for better performance

If you have a Nvidia GPU, you can benefit from a huge performance boost. 
But it comes at the cost of needing to install complex stuff. You will need:

- Nvidia drivers able to run CUDA 10.0 or higher (that means, at least drivers version 410.48).
- CUDA 10.0 or higher.

If you have both, then you can install the extra python dependencies from `gpu_requirements.txt`, and use the `--gpu` parameter with `viba.py`.
