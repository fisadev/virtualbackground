import json
import logging
from asyncio import run, get_running_loop, sleep, gather
from pathlib import Path

import cv2
import numpy as np
import pyfakewebcam
import requests
import tensorflow as tf
from tfjs_graph_converter.api import load_graph_model


FORMAT = '%(asctime)-15s|%(levelname)s|%(name)s|%(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger('viba')


def download_file(url, path):
    """
    Download a file from a url to a path in disk.
    """
    response = requests.get(url)
    with open(path, 'wb') as destination_file:
        destination_file.write(response.content)


class Cam:
    """
    A camera, either real (to read frames from), or fake (to send frames to).
    """
    def __init__(self, device, size, fps, is_real=True):
        self.device = device
        self.size = size
        self.fps = fps
        self.is_real = is_real

        if self.is_real:
            self.init_real_cam()
        else:
            self.init_fake_cam()

    def init_real_cam(self):
        """
        Initialize a real camera, using opencv to capture frames from it.
        """
        capturer = cv2.VideoCapture(self.device)
        capturer.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capturer.set(cv2.CAP_PROP_FPS, self.fps)

        self.interface = capturer

    def init_fake_cam(self):
        """
        Initialize a fake camera, using pyfakewebcam to send frames to it.
        """
        self.interface = pyfakewebcam.FakeWebcam(self.device, self.width, self.height)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    async def read_frame(self):
        """
        Read a real frame from a real camera.
        """
        assert self.is_real
        frame = None
        while frame is None:
            await sleep(0)
            _, frame = self.interface.read()

        # why would anyone use BGR??
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    async def write_frame(self, frame):
        """
        Write a fake frame to be displayed in a fake camera.
        """
        assert not self.is_real

        frame_height, frame_width, _ = frame.shape
        if (frame_width, frame_height) != self.size:
            frame = cv2.resize(frame, self.size)

        await sleep(0)

        self.interface.schedule_frame(frame)


class SegmenterModel:
    """
    A model able to segment people in images, using tensorflow.
    Currently based on mobilenet.
    """
    BASE_MODELS_PATH = Path('./models/')
    BASE_MODELS_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/{}'

    KNOWN_MODELS_URLS = {
        'mobilenet_quant4_100_stride16': 'bodypix/mobilenet/float/100/model-stride16.json',
        'mobilenet_quant4_075_stride16': 'bodypix/mobilenet/float/075/model-stride16.json',
        # there are plenty of other models, but this project has been tested and tuned up for these
        # you can try others if you want :)
    }

    def __init__(self, model_name, segmentation_threshold, use_gpu):
        if model_name not in self.KNOWN_MODELS_URLS:
            raise ValueError("Uknown model: {}".format(model_name))

        self.model_name = model_name
        self.segmentation_threshold = segmentation_threshold
        self.use_gpu = use_gpu

        self.init_graph()

    def init_graph(self):
        """
        Get a model from the published tensorflowjs models. If it's not present, download it.
        """
        self.download_tfjs_model()
        self.graph = load_graph_model(str(self.model_path))

    @property
    def model_path(self):
        """
        Build path to the specified model.
        """
        return self.BASE_MODELS_PATH / self.model_name / 'model.json'

    def download_tfjs_model(self):
        """
        If the model from the published tensorflowjs models isn't present, download it.
        """
        model_dir_path = self.model_path.parent
        model_dir_path.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            logger.info("TensorflowJS model not present, will download it")

            logger.info("Downloading model definition...")
            model_url = self.BASE_MODELS_URL.format(self.KNOWN_MODELS_URLS[self.model_name])
            download_file(
                model_url,
                self.model_path,
            )

            parent_model_url = '/'.join(model_url.split('/')[:-1]) + '/{}'
            definition = json.loads(self.model_path.read_text())
            for weights_manifest in definition['weightsManifest']:
                for weights_file_name in weights_manifest['paths']:
                    logger.info("Downloading weights: %s...", weights_file_name)
                    download_file(
                        parent_model_url.format(weights_file_name),
                        model_dir_path / weights_file_name,
                    )

    def apply_model(self, inputs):
        """
        Apply the tensorflow model to get the segmentation mask outputs.
        """
        with tf.compat.v1.Session(graph=self.graph) as sess:
            with tf.device("/gpu:0" if self.use_gpu else "/cpu:0"):
                input_tensor = self.graph.get_tensor_by_name('sub_2:0')
                results = sess.run(['float_segments:0'], feed_dict={input_tensor: inputs})

                segments = np.squeeze(results[0], 0)

            segment_scores = tf.sigmoid(segments)
            mask = tf.math.greater(segment_scores, tf.constant(self.segmentation_threshold))
            segmentation_mask = mask.eval()

            segmentation_mask = np.reshape(
                segmentation_mask, (segmentation_mask.shape[0], segmentation_mask.shape[1])
            ).astype(np.uint8)

            return segmentation_mask

    async def get_mask(self, image):
        """
        Get the mask that defines the area occupied by people in the image.
        """
        original_height, original_width, _ = image.shape

        image = (image / 127.) - 1
        image_as_inputs = np.expand_dims(image, 0)

        await sleep(0)

        loop = get_running_loop()
        segmentation_mask = await loop.run_in_executor(None, self.apply_model, image_as_inputs)
        segmentation_mask = cv2.resize(segmentation_mask, (original_width, original_height))

        return segmentation_mask


class VirtualBackground:
    """
    A virtual background achieved by reading frames from a real camera, finding the location of
    people in the image, replacing the background with a custom image, and then sending the
    modified frame to a fake camera.
    """
    def __init__(self, model, real_cam, fake_cam, background_path):
        self.model = model
        self.real_cam = real_cam
        self.fake_cam = fake_cam

        self.last_frame = None
        self.current_mask = None

        self.load_background(background_path)

    def load_background(self, background_path):
        """
        Load and prepare the background to use in the frames.
        """
        background = cv2.imread(background_path)
        background = cv2.resize(background, self.real_cam.size)
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    async def frames_loop(self):
        """
        Get new frames, apply the current mask and schedule them as fast as possible.
        """
        logger.info("Started frames loop")
        while True:
            logger.debug("Reading a new frame...")
            frame = await self.real_cam.read_frame()
            self.last_frame = frame

            logger.debug("Enhancing the frame...")
            if self.current_mask is None:
                enhanced_frame = frame
            else:
                enhanced_frame = await self.enhance_frame(frame)

            logger.debug("Sending the enhanced frame to the fake cam...")
            await self.fake_cam.write_frame(enhanced_frame)

    async def mask_loop(self):
        """
        Get and refresh the mask as fast as possible.
        """
        logger.info("Started mask loop")
        while True:
            if self.last_frame is None:
                await sleep(0)
                continue

            logger.debug("Updating mask...")
            raw_mask = await self.model.get_mask(self.last_frame)
            self.current_mask = await self.post_process_mask(raw_mask)

    async def enhance_frame(self, frame):
        """
        Given a frame, enhance it using the current mask, and any optional effects.
        """
        frame = frame.copy()
        for channel in range(frame.shape[2]):
            frame[:, :, channel] = (
                frame[:, :, channel] * self.current_mask
                + self.background[:, :, channel] * (1 - self.current_mask)
            )
            await sleep(0)

        return frame

    async def post_process_mask(self, mask):
        """
        Dilate and blur the mask, so the limits between the person and background aren't that
        abrupt and visible.
        """
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=2)
        await sleep(0)
        mask = cv2.blur(mask.astype(float), (30, 30))
        return mask

    def run(self):
        """
        Run the virtual camera.
        """
        async def main_loop():
            """
            The main loop just stays alive while the frames and mask loops are alive.
            """
            await gather(self.frames_loop(), self.mask_loop())

        run(main_loop())


def main():
    """
    Cli script.
    """
    viba = VirtualBackground(
        model=SegmenterModel(
            model_name='mobilenet_quant4_100_stride16',
            segmentation_threshold=0.7,
            use_gpu=False,
        ),
        real_cam=Cam(device="/dev/video0", size=(640, 480), fps=30, is_real=True),
        fake_cam=Cam(device="/dev/video20", size=(640, 480), fps=30, is_real=False),
        background_path='./sample_background.jpg',
    )
    viba.run()


if __name__ == '__main__':
    main()
