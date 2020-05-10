# ViBa: virtual backgrounds

Fake webcam that replaces your background with a custom image of your choice.

This is a **work in progress**, although it does work if you have all the right dependencies installed.

# Installation

### 1. Fake webcam device

You will need to do some weird stuff first: adding a kernel module to be able to have "fake" webcams.
This is done using the `v4l2loopback-dkms` package. On Debian or Ubuntu, you can install and configure it like this:

```bash
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
```

This means that from now on, you will have a `/dev/video20` device that simulates being a webcam.

### 2. Python libs

Then you need to install the dependencies in the `requirements.txt` file, as usual with python. 
A virtualenv is recommended:

```bash
    # create and activate the virtualenv
    python3.7 -m venv my_viba_venv
    source my_viba_venv/bin/activate

    # upgrade pip, and then install the dependencies
    pip install pip --upgrade
    pip install -r requirements.txt
```


# Usage

Just run the `viba.py` script, with your virtualenv activated:

```bash
    # activate viba's virtualenv
    source my_viba_venv/bin/activate

    # run viba
    python viba.py
```

The script allows for a number of params to customize how the virtual background works:

```man
Options:
  --background TEXT               The background image to use in the webcam.
  --use-gpu                       Force the use of a CUDA enabled GPU, to
                                  improve performance. Remember that this has
                                  extra dependencies, more info in the README.

  --real-cam-resolution <INTEGER INTEGER>...
                                  The resolution of the real webcam. We highly
                                  recommend using a small value because of
                                  performance reasons, specially if you aren't
                                  using a high end GPU with viba. The value
                                  must be a tuple with the structure: (width,
                                  height). Example: --real-cam-resolution 640
                                  480

  --fake-cam-resolution <INTEGER INTEGER>...
                                  The resolution of the fake webcam. We
                                  recommend using a small value because of
                                  performance reasons, but this isn't as
                                  important as the real cam resolution. Also,
                                  useful info: some web conference services
                                  like Jitsi ignore webcams bellow 720p. The
                                  value must be a tuple with the structure:
                                  (width, height). Example: --fake-cam-
                                  resolution 640 480

  --real-cam-fps INTEGER          The speed (frames per second) of the real
                                  webcam.

  --real-cam-device TEXT          The linux device in which the real cam
                                  exists.

  --fake-cam-device TEXT          The linux device in which the fake cam
                                  exists (the one created using v4l2loopback.

  --model-name [mobilenet_quant4_100_stride16|mobilenet_quant4_075_stride16]
                                  The tensorflowjs model that will be used to
                                  detect people in the video. If you have
                                  trouble with performance, you can try using
                                  'mobilenet_quant4_075_stride16', which is a
                                  little bit faster.

  --segmentation-threshold FLOAT RANGE
                                  How much of the image will be considered as
                                  a 'person'. A lower value means less
                                  confidence required, so more regions will be
                                  considered as a 'person'. A higher value
                                  means the opposite. Must be a value between
                                  0 and 1.

  --debug                         Debug mode: print a lot of extra text during
                                  execution, to debug issues.
```

# Usage as a lib

You can also use it from your own programs. Just take a look at the `main` function in `viba.py`, that's a perfect example and very simple.

# SPEEEED: using GPU for better performance

If you have a Nvidia GPU, you can benefit from a huge performance boost. 
But it comes at the cost of needing to install complex stuff. You will need:

- Nvidia drivers able to run CUDA 10.0 or higher (that means, at least drivers version 410.48).
- CUDA 10.0 or higher.

If you have both, then you can install the extra python dependencies from `gpu_requirements.txt`, and use the `--gpu` parameter with `viba.py`.
