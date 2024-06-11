import torch
import cv2
import pandas as pd
import numpy as np

from pog import Logger
from pig import ConfigRead
from urllib.request import Request, urlopen
from warnings import simplefilter
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

current_dir = Path().resolve()
root_dir = next((parent for parent in current_dir.parents if parent.name == 'content'), None)

logger = Logger(root_dir).get_logger('pom')
logger.info(f'Logger from POM module')
simplefilter("ignore")


class ModelInference:
    def __init__(self, package_path, photo_url_list):
        self.config = ConfigRead(package_path)
        self.package_path = package_path
        self.header = {'User-Agent': self.config.parse_value('header', 'User-Agent'),
                       'Referer': self.config.parse_value('header', 'Referer'),
                       'Accept': self.config.parse_value('header', 'Accept')}
        self.image_sharpness_threshold = float(self.config.parse_value('image_processing', 'image_sharpness'))
        self.model_a_path = Path(self.package_path, 'custom_models/', 'model_a.pt')
        self.model_a_classes = [int(x) for x in self.config.parse_list('model_a', 'model_a_classes')]
        self.model_a_confidence = float(self.config.parse_value('model_a', 'model_confidence'))
        self.model_a_iou = float(self.config.parse_value('model_a', 'model_iou'))
        self.model_a_keys = set(self.config.parse_list('container', 'model_a_container'))
        self.model_a_container = self.config.parse_dict('container', 'model_a_container')
        self.model_b_path = Path(self.package_path, 'custom_models/', 'model_b.pt')
        self.model_b_classes = [int(x) for x in self.config.parse_list('model_b', 'model_b_classes')]
        self.model_b_confidence = float(self.config.parse_value('model_b', 'model_confidence'))
        self.model_b_iou = float(self.config.parse_value('model_a', 'model_iou'))
        self.model_b_keys = set(self.config.parse_list('container', 'model_b_container'))
        self.model_b_container = self.config.parse_dict('container', 'model_b_container')
        self.main_container = self.config.parse_dict('container', 'main_container')
        self.new_col_dict = dict(zip(self.config.parse_list('container', 'main_container'),
                                     self.config.parse_list('container', 'rename_container')))
        self.device = 'cpu'
        self.model_a_loaded = None  # Product brands
        self.model_b_loaded = None  # product pack sizes
        self.model_a_data = None
        self.model_b_data = None
        self.photo_url_list = photo_url_list
        self.main_data = None
        self.lock_thread = None
        self.thread_output = None

    def read_image_url(self, arg):
        filename, img_RGB, img_quality = arg, None, "Unidentified"
        try:
            if str(arg).startswith('http'):
                request = Request(arg, headers=self.header)
                response = urlopen(request, timeout=100)

                if response.getcode() == 200:
                    img_arr_read = np.asarray(bytearray(response.read()), dtype=np.uint8)
                    img_BGR = cv2.imdecode(img_arr_read, cv2.IMREAD_COLOR)
                    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                    img_quality = self.image_quality(arg, img_RGB)
                    return filename, img_RGB, img_quality
            else:
                img_BGR = cv2.imread(arg, cv2.IMREAD_COLOR)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                img_quality = self.image_quality(arg, img_RGB)
                return filename, img_RGB, img_quality

        except Exception as err:
            logger.error(f'ModelInference | read_image_url | {filename} | {err.__class__.__name__} {err}')

        finally:
            return filename, img_RGB, img_quality

    @staticmethod
    def image_slicing(img):
        sliced_img = []

        # Get the dimensions of the image.
        height, width, channels = img.shape

        # Define slice size
        slice_size = (height // 3, width // 3)

        # Slice the image into 6 images.
        for i in range(3):
            for j in range(3):
                sliced_img.append(img[i * slice_size[0]:(i + 1) * slice_size[0],
                                  j * slice_size[1]:(j + 1) * slice_size[1]])
        return sliced_img

    @staticmethod
    def image_sharpness(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """local_perceived_sharpness"""
        laplacian_sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return laplacian_sharpness

    def image_quality(self, img_name, img_rgb):
        try:
            sliced_img = self.image_slicing(img_rgb)
            img_sharpness = [self.image_sharpness(x) for x in sliced_img]
            lap_mean = np.mean(img_sharpness)

            if lap_mean <= self.image_sharpness_threshold:
                return "Bad quality"
            else:
                return "Accepted quality"

        except Exception as err:
            logger.error(f'ModelInference | image_quality | {Path(img_name).name} | {err.__class__.__name__} {err}')

    def load_model_a_pretrained(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model_a_loaded = torch.hub.load(str(self.package_path), 'custom',
                                             path=self.model_a_path,
                                             device=self.device,
                                             source='local',
                                             force_reload=False,
                                             _verbose=False)
        self.model_a_loaded.multi_label = False  # NMS multiple labels per box
        self.model_a_loaded.conf = self.model_a_confidence  # NMS confidence threshold
        self.model_a_loaded.iou = self.model_a_iou  # NMS IoU threshold
        self.model_a_loaded.classes = self.model_a_classes
        self.model_a_loaded.agnostic = False  # NMS class-agnostic
        self.model_a_loaded.amp = False  # Automatic Mixed Precision (AMP) inference
        self.model_a_loaded.max_det = 1000  # maximum number of detections per image

        logger.info(f'ModelInference | Pretrained model A was loaded, classes: {self.model_a_classes}')

    def load_model_b_pretrained(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model_b_loaded = torch.hub.load(str(self.package_path), 'custom',
                                             path=self.model_b_path,
                                             device=self.device,
                                             source='local',
                                             force_reload=False,
                                             _verbose=False)
        self.model_b_loaded.multi_label = False  # NMS multiple labels per box
        self.model_b_loaded.conf = self.model_b_confidence  # NMS confidence threshold
        self.model_b_loaded.iou = self.model_b_iou  # NMS IoU threshold
        self.model_b_loaded.classes = self.model_b_classes
        self.model_b_loaded.agnostic = False  # NMS class-agnostic
        self.model_b_loaded.amp = False  # Automatic Mixed Precision (AMP) inference
        self.model_b_loaded.max_det = 1000  # maximum number of detections per image

        logger.info(f'ModelInference | Pretrained model B was loaded, classes: {self.model_b_classes}')

    def model_a_inference(self, f_name, img):
        self.model_a_data = self.model_a_container
        self.model_a_data['filename'] = f_name
        self.model_a_data['B_Status'] = 'N'
        try:
            brand_infer = self.model_a_loaded(img)
            brand_detection = brand_infer.extract()
            brand_detection['filename'] = f_name
            brand_detection['B_Status'] = 'Y'

            brand_detection_f = {key: brand_detection[key] for key in brand_detection.keys() & self.model_a_keys}
            self.model_a_data = {**self.model_a_data, **brand_detection_f}

        except Exception as err:
            logger.error(f'ModelInference | model_a_inference: {err.__class__.__name__} {err}')

        finally:
            return self.model_a_data

    def model_b_inference(self, f_name, img):
        self.model_b_data = self.model_b_container
        self.model_b_data['filename'] = f_name
        self.model_b_data['S_Status'] = 'Y'
        try:
            size_infer = self.model_b_loaded(img)
            size_detection = size_infer.extract()
            size_detection['filename'] = f_name
            size_detection['S_Status'] = 'Y'

            size_detection_f = {key: size_detection[key] for key in size_detection.keys() & self.model_b_keys}
            self.model_b_data = {**self.model_b_data, **size_detection_f}

        except Exception as err:
            logger.error(f'ModelInference | model_b_inference: {err.__class__.__name__} {err}')

        finally:
            return self.model_b_data

    def inference_image(self, arg, lock):
        with lock:
            self.main_data = self.main_container

            # Read image from URL: return filename, img_RGB, img_quality
            filename, img_RGB, img_quality = self.read_image_url(arg)
            self.main_data['filename'] = filename
            self.main_data['Photo_Quality'] = img_quality

            if img_RGB is not None:
                # Update url_status to main_container
                self.main_data['URL_Status'] = 'success'

                # Process Model A Inference: return self.model_a_data
                self.model_a_inference(filename, img_RGB)

                # Update model A data to main_container
                self.main_data = {**self.main_data, **self.model_a_data}

                # Process Model B Inference: return self.model_b_data
                self.model_b_inference(filename, img_RGB)

                # Update model B data to main_container
                self.main_data = {**self.main_data, **self.model_b_data}

            else:
                self.main_data['URL_Status'] = 'error'
                self.main_data['B_Status'] = 'N'
                self.main_data['S_Status'] = 'N'

            return self.main_data

    def ThreadPoolExecutor_submit(self, func, iterable):
        t1 = perf_counter()
        logger.info(f'ModelInference | Thread is starting')

        self.lock_thread = Lock()
        self.thread_output = pd.DataFrame()
        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(func, i, self.lock_thread) for i in iterable]
                for f in as_completed(futures):
                    df = pd.DataFrame([f.result()])
                    self.thread_output = pd.concat([self.thread_output, df], ignore_index=True)
                logger.info(f'ModelInference | Thread was completed in {perf_counter() - t1:.2f}')

        except Exception as err:
            logger.error(f'ModelInference | Thread error: {err.__class__.__name__} {err}')

        finally:
            return self.thread_output

    def inference_image_process(self):
        try:
            self.load_model_a_pretrained()
            self.load_model_b_pretrained()
            detection_data = self.ThreadPoolExecutor_submit(self.inference_image, self.photo_url_list)
            detection_data.rename(columns=self.new_col_dict, inplace=True)
            return detection_data

        except Exception as err:
            logger.error(f'ERROR: ModelInference | inference_process: {err.__class__.__name__} {err}')
