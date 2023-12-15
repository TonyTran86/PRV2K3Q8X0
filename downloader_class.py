import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from urllib.request import urlopen, Request, urlretrieve
from warnings import simplefilter
from image_process import image_processing
from logger_class import Logger

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/55.0.2883.87 Safari/537.36', 'Referer': 'HTTPS://PEPSICODMS.COM',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}

# Set source path --------------------------------
root_dir = Path().resolve()
logger = Logger().get_logger('URLDownloader_class')
logger.info('Logger from downloader_class.py')
simplefilter("ignore")


class URLDownloader:
    def __init__(self):
        self.url_parse = None
        self.url_read = None

    def parse_image_url(self, arg):
        try:
            request = Request(arg, headers=headers)
            response = urlopen(request, timeout=90)

            if response.getcode() == 200:
                # Parse image's url and encode to np.ndarray
                self.url_parse = np.asarray(bytearray(response.read()), dtype=np.uint8)
                self.url_read = cv2.imdecode(self.url_read, cv2.IMREAD_COLOR)
                return self.url_read
            else:
                return None

        except Exception as err:
            URLDownloader.print_out(f'parse_image_url error: {arg} {err.__class__.__name__} {err}', mode='error')
            pass

    @staticmethod
    def print_out(input_string, mode='debug'):  # debug, error, info
        print(input_string)
        if mode == 'error':
            logger.error(input_string)
        else:
            logger.debug(input_string)


    def download_url(self, arg):
        img_url, img_name, img_status, img_download, img_location = arg[0], arg[1], arg[2], arg[3], arg[4]
        try:
            request = Request(img_url, headers=headers)
            response = urlopen(request, timeout=60)

            if response.getcode() == 200:
                # Image processing
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # img_rotated = image_process.rotate90_img(img)
                height, width, channels = img.shape
                img_quality = image_processing(img)

                f_folder = self.folder_name[0]
                f_path = Path(self.download_folder, img_name)

                if height < 640 or width < 640:
                    f_folder = self.folder_name[2]
                    f_path = Path(self.low_size, img_name)

                if not img_quality:
                    f_folder = self.folder_name[1]
                    f_path = Path(self.bad_quality, img_name)

                if not f_path.is_file():
                    urlretrieve(img_url, f_path)

                    self.lst_cache.append([img_url, img_name, 'valid', 'downloaded', f_folder, img_quality])
            else:
                self.lst_cache.append([img_url, img_name, 'invalid', 'invalid', 'invalid', 'invalid'])

        except Exception as err:
            self.lst_cache.append([img_url, img_name, 'error', 'error', 'error', 'error'])
            print(f'Image processing error: {arg} {err.__class__.__name__} {err}')
            pass
