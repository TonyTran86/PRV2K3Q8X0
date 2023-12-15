import cv2
import numpy as np
import pandas as pd
import torch
from os import path
from time import time
from warnings import simplefilter
from pathlib import Path
from urllib.request import urlopen, Request
from thread_executor_class import ThreadExecutor
from logger_class import Logger

# Set source path --------------------------------
root_dir = Path().resolve()
logger = Logger().get_logger('InferenceModel_class')
logger.info('Logger from inference_class.py')
simplefilter("ignore")

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/55.0.2883.87 Safari/537.36', 'Referer': 'HTTPS://PEPSICODMS.COM',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


class InferenceModel:
    def __init__(self):
        self.model_a_loaded = None  # Product brands
        self.model_b_loaded = None  # product pack sizes
        self.save_path = None
        self.model_a_classes = []
        self.model_b_classes = []
        self.load_pretrained_model = None

        self.local_repo = Path(root_dir, 'repo_master')
        self.model_a_pretrained = Path(root_dir, 'repo_master/custom_models/model_a.pt')
        self.model_b_pretrained = Path(root_dir, 'repo_master/custom_models/model_b.pt')
        self.device = 'cpu'
        self.img_arr_read = None
        self.imgRBG_read = None
        self.main_container = None
        self.main_container_a = None
        self.main_container_b = None
        self.model_a_data = None
        self.model_b_data = None
        self.pattern = None
        self.detection_data = None
        self.re_detection_data = None

    def load_model_a_pretrained(self, class_list, confidence=0.0, iou=0.0):
        if torch.cuda.is_available():
            self.device = 'cuda'

        for c in class_list:
            self.model_a_classes.append(c)
        InferenceModel.print_out(f'INFO: List of model A classes | {self.model_a_classes}')

        self.model_a_loaded = torch.hub.load(str(self.local_repo), 'custom',
                                             path=self.model_a_pretrained,
                                             device='cpu',
                                             source='local',
                                             force_reload=False,
                                             _verbose=False)
        self.model_a_loaded.multi_label = False  # NMS multiple labels per box
        self.model_a_loaded.conf = confidence  # NMS confidence threshold
        self.model_a_loaded.iou = iou  # NMS IoU threshold
        self.model_a_loaded.classes = self.model_a_classes
        self.model_a_loaded.agnostic = False  # NMS class-agnostic
        self.model_a_loaded.amp = False  # Automatic Mixed Precision (AMP) inference
        self.model_a_loaded.max_det = 1000  # maximum number of detections per image

        InferenceModel.print_out(f'Pretrained model A was loaded')

    def load_model_b_pretrained(self, class_list, confidence=0.0, iou=0.0):
        if torch.cuda.is_available():
            self.device = 'cuda'

        for c in class_list:
            self.model_b_classes.append(c)
        InferenceModel.print_out(f'INFO: List of model B classes | {self.model_b_classes}')

        self.model_b_loaded = torch.hub.load(str(self.local_repo), 'custom',
                                             path=self.model_b_pretrained,
                                             device='cpu',
                                             source='local',
                                             force_reload=False,
                                             _verbose=False)
        self.model_b_loaded.multi_label = False  # NMS multiple labels per box
        self.model_b_loaded.conf = confidence  # NMS confidence threshold
        self.model_b_loaded.iou = iou  # NMS IoU threshold
        self.model_b_loaded.classes = self.model_b_classes
        self.model_b_loaded.agnostic = False  # NMS class-agnostic
        self.model_b_loaded.amp = False  # Automatic Mixed Precision (AMP) inference
        self.model_b_loaded.max_det = 1000  # maximum number of detections per image

        InferenceModel.print_out(f'Pretrained model B was loaded')

    # @staticmethod
    def url_parse(self, arg):
        try:
            request = Request(arg, headers=headers)
            response = urlopen(request, timeout=100)
            if response.getcode() == 200:
                self.img_arr_read = np.asarray(bytearray(response.read()), dtype=np.uint8)
            else:
                self.img_arr_read = None

        except Exception as err:
            InferenceModel.print_out(f'ERROR: {self.name_pattern_change(arg)} | URL parse error: {err.__class__.__name__} {err}',
                                     mode='error')
            self.img_arr_read = None

        finally:
            return self.img_arr_read

    @staticmethod
    def arg_parser(arg, file_type=None):
        try:
            if Path(arg).is_dir():
                files = [x for x in Path.iterdir(Path(arg)) if x.name.endswith('.' + file_type)]
                return files
            else:
                return arg

        except Exception as err:
            InferenceModel.print_out(f'ERROR: arg_parser error: {err.__class__.__name__} {err}', mode='error')
            return arg

    # @staticmethod
    def image_read(self, arg):
        try:
            if str(arg).startswith('https://') or str(arg).startswith('http://'):
                self.url_parse(arg)

                if self.img_arr_read is not None:
                    img_arr = np.asarray(bytearray(self.img_arr_read), dtype=np.uint8)
                    img_BGR = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    self.imgRBG_read = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                    # img_quality = image_process.image_processing(img_RGB)
                else:
                    self.imgRBG_read = None
            else:
                img_BGR = cv2.imread(arg, cv2.IMREAD_COLOR)
                self.imgRBG_read = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                # img_quality = image_process.image_processing(img_RGB)

        except Exception as err:
            InferenceModel.print_out(f'ERROR: {self.name_pattern_change(arg)} | img_read URL error: {err.__class__.__name__} {err}',
                                     mode='error')
            self.imgRBG_read = None

        finally:
            return self.imgRBG_read

    def model_a_inference(self, img_input):
        self.model_a_data = {'a_status': 'error', 'poca': 0, 'lays': 0, 'laysmax': 0, 'doritos': 0}
        try:
            brand_infer = self.model_a_loaded(img_input)
            brand_detection = brand_infer.extract()
            brand_detection['a_status'] = 'success'
            brand_detection = {k: v for k, v in brand_detection.items() if v}

            if 'poca' not in brand_detection:
                brand_detection['poca'] = 0

            if 'lays' not in brand_detection:
                brand_detection['lays'] = 0

            if 'laysmax' not in brand_detection:
                brand_detection['laysmax'] = 0

            if 'doritos' not in brand_detection:
                brand_detection['doritos'] = 0

            if 'stax' not in brand_detection:
                brand_detection['stax'] = 0

            self.model_a_data = {**self.model_a_data, **brand_detection}

        except Exception as err:
            InferenceModel.print_out(
                f'ERROR: inference_img model A error: {err.__class__.__name__} {err}',
                mode='error')
        finally:
            return self.model_a_data

    def model_b_inference(self, img_input):
        self.model_b_data = {'b_status': 'error', 'lays10': 0, 'nutz': 0}
        try:
            size_infer = self.model_b_loaded(img_input)
            size_detection = size_infer.extract()
            size_detection['b_status'] = 'success'
            size_detection = {k: v for k, v in size_detection.items() if v}

            if 'lays10' not in size_detection:
                size_detection['lays10'] = 0

            if 'nutz' not in size_detection:
                size_detection['nutz'] = 0

            self.model_b_data = {**self.model_b_data, **size_detection}

        except Exception as err:
            InferenceModel.print_out(
                f'ERROR: inference_img model B error: {err.__class__.__name__} {err}',
                mode='error')
        finally:
            return self.model_b_data

    def name_pattern_change(self, input_string, mode='name'):
        self.pattern = 'imageName='
        pattern_index = str(input_string).rfind(self.pattern)

        if mode == 'path':
            return input_string
        else:
            if pattern_index < 0:
                return path.basename(input_string)
            else:
                return input_string[int(pattern_index) + len(self.pattern):]

    def inference_image(self, arg):
        # Re-construct file name
        f_name = self.name_pattern_change(arg, mode='path')

        # Build main data
        self.main_container = {'filename': None, 'url_status': None, 'a_status': None, 'b_status': None,
                               'poca': 0, 'lays': 0, 'laysmax': 0, 'doritos': 0, 'lays10': 0, 'nutz': 0}

        # Read image from URL
        self.image_read(arg)
        self.main_container['filename'] = f_name

        if self.imgRBG_read is not None:
            self.main_container['url_status'] = 'success'

            # Inference Model A
            self.model_a_inference(self.imgRBG_read)
            self.model_a_data['filename'] = f_name
            self.main_container = {**self.main_container, **self.model_a_data}

            # Inference Model B
            self.model_b_inference(self.imgRBG_read)
            self.model_b_data['filename'] = f_name
            self.main_container = {**self.main_container, **self.model_b_data}

        else:
            self.main_container['url_status'] = 'error'

        return self.main_container

    def inference_model_a(self, arg):
        # Re-construct file name
        f_name = self.name_pattern_change(arg)

        # Build main data
        self.main_container_a = {'filename': None, 'url_status': None, 'a_status': None, 'poca': 0, 'lays': 0,
                                 'laysmax': 0, 'doritos': 0}

        # Read image from URL
        self.image_read(arg)
        self.main_container_a['filename'] = f_name

        if self.imgRBG_read is not None:
            self.main_container_a['url_status'] = 'success'

            # Inference Model A
            self.model_a_inference(self.imgRBG_read)
            self.model_a_data['filename'] = f_name
            self.main_container_a = {**self.main_container_a, **self.model_a_data}

        else:
            self.main_container_a['url_status'] = 'error'

        return self.main_container_a

    def inference_model_b(self, arg):
        # Re-construct file name
        f_name = self.name_pattern_change(arg)

        # Build main data
        self.main_container_b = {'filename': None, 'url_status': None, 'b_status': None, 'lays10': 0, 'nutz': 0}

        # Read image from URL
        self.image_read(arg)
        self.main_container_b['filename'] = f_name

        if self.imgRBG_read is not None:
            self.main_container_b['url_status'] = 'success'

            # Inference Model B
            self.model_b_inference(self.imgRBG_read)
            self.model_b_data['filename'] = f_name
            self.main_container_b = {**self.main_container_b, **self.model_b_data}

        else:
            self.main_container_b['url_status'] = 'error'

        return self.main_container_b

    def inference_update_process(self, model=None):
        if model is not None:
            model_string = str(model).lower()
            col_name = model_string + "_status"
            model_err_check = 'error' in self.detection_data[col_name].values

            if model_err_check:
                while True:
                    model_err_data = self.detection_data[(self.detection_data[col_name] == 'error')]
                    err_cnt = model_err_data.shape[0]

                    if not model_err_data.empty:
                        InferenceModel.print_out(f'INFO: Model A detection error found {err_cnt}')
                        f_url = model_err_data['filename']

                        if model_string == 'a':
                            self.re_detection_data = ThreadExecutor.cf_ThreadPoolExecutor_submit(self.inference_model_a, f_url)
                        else:
                            self.re_detection_data = ThreadExecutor.cf_ThreadPoolExecutor_submit(self.inference_model_b, f_url)

                        self.detection_data['filenamex'] = self.detection_data['filename'].map(str)
                        self.detection_data.set_index('filenamex', inplace=True)
                        self.detection_data.update(self.re_detection_data.set_index('filename'))
                        self.detection_data.reset_index(inplace=True)  # to recover the initial structure
                        self.detection_data = self.detection_data.drop(['filenamex'], axis=1, errors='ignore')
                        InferenceModel.print_out(f'INFO: inference_process | Model {str(model).upper()} re-detection: {err_cnt}')
        else:
            pass

        return self.detection_data

    def inference_process(self, photo_url_list, debug=False):
        process_start = time()

        try:
            model_a_classes = [0, 1, 3, 4, 5]
            self.load_model_a_pretrained(model_a_classes, confidence=0.690, iou=0.5)

            model_b_classes = [0, 1]
            self.load_model_b_pretrained(model_b_classes, confidence=0.819, iou=0.5)

            self.detection_data = ThreadExecutor.cf_ThreadPoolExecutor_submit(self.inference_image, photo_url_list)

            if debug:
                self.detection_data.to_excel('detection_data.xlsx', index=False)

            # model_b_err_check = 'error' in detection_data['b_status'].values
            # while True:
            #     model_a_err_data = detection_data[(detection_data['a_status'] == 'error')]
            #     model_b_err_data = detection_data[(detection_data['b_status'] == 'error')]
            #     a_cnt = model_a_err_data.shape[0]
            #     b_cnt = model_b_err_data.shape[0]
            #
            #     if not model_a_err_data.empty:
            #         InferenceModel.print_out(f'INFO: Model A detection error found {a_cnt}')
            #         a_url = model_a_err_data['filename']
            #         new_data = ThreadExecutor.cf_ThreadPoolExecutor_submit(self.inference_image, a_url)
            #
            #         detection_data['filenamex'] = detection_data['filename'].map(str)
            #         detection_data.set_index('filenamex', inplace=True)
            #         detection_data.update(new_data.set_index('filename'))
            #         detection_data.reset_index(inplace=True)  # to recover the initial structure
            #         detection_data = detection_data.drop(['filenamex'], axis=1, errors='ignore')
            #
            #         InferenceModel.print_out(f'INFO: inference_process | Model A re-detection: {a_cnt}')
            #     else:
            #         if not model_b_err_data.empty:
            #             InferenceModel.print_out(f'INFO: Model B detection error found {b_cnt}')
            #             b_url = model_b_err_data['filename']
            #             new_data = ThreadExecutor.cf_ThreadPoolExecutor_submit(self.inference_image, b_url)
            #
            #             detection_data['filenamex'] = detection_data['filename'].map(str)
            #             detection_data.set_index('filenamex', inplace=True)
            #             detection_data.update(new_data.set_index('filename'))
            #             detection_data.reset_index(inplace=True)  # to recover the initial structure
            #             detection_data = detection_data.drop(['filenamex'], axis=1, errors='ignore')
            #
            #             InferenceModel.print_out(f'INFO: inference_process | Model B re-detection: {b_cnt}')
            #         else:
            #             break

            InferenceModel.print_out(f'INFO: inference_process in | {time() - process_start}')

            return self.detection_data

        except Exception as err:
            InferenceModel.print_out(f'ERROR: inference_process | {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def print_out(input_string, mode='debug'):  # debug, error, info
        print(input_string)
        if mode == 'error':
            logger.error(input_string)
        else:
            logger.debug(input_string)


if __name__ == '__main__':
    pass
