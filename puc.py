import pandas as pd
import numpy as np
import shutil

from warnings import simplefilter
from pog import Logger
from pathlib import Path
from sim import MainApp, pack_report_file
from pig import ConfigRead
from pom import ModelInference

current_dir = Path().resolve()
root_dir = next((parent for parent in current_dir.parents if parent.name == 'content'), None)

logger = Logger(root_dir).get_logger('puc')
logger.info(f'Logger from PUC module')
simplefilter("ignore")


class ProcessReport:
    def __init__(self, input_id, input_pw, role, func, packed_file):
        self.input_id = input_id
        self.input_pw = input_pw
        self.role = role
        self.func = func
        (self.main_data, self.package_path,
         self.packed_file, self.packed_result_path) = MainApp(self.input_id,
                                                              self.input_pw,
                                                              self.role,
                                                              self.func,
                                                              packed_file).load_data()
        self.config = ConfigRead(self.package_path)
        self.result_template = self.config.parse_list('result_template', 'result_template')
        self.tt_result_template = self.config.parse_list('result_template', 'tt_result_template') + self.result_template
        self.ot_result_template = self.config.parse_list('result_template', 'ot_result_template') + self.result_template
        self.arc_file_path = Path(self.packed_result_path, Path(self.packed_file).name)
        self.xlsx_file_name = Path(self.packed_file).stem + '_result.xlsx'
        self.photo_url_list = None
        self.detection_data = None

    def read_report_data(self):
        try:
            if self.role == 'TT' and self.main_data.columns.to_list() != self.tt_result_template:
                logger.warning(f'ProcessReport | read_report_data | Invalid Report file template')
            elif self.role == 'OT' and self.main_data.columns.to_list() != self.ot_result_template:
                logger.warning(f'ProcessReport | read_report_data | Invalid Report file template')
            else:
                self.photo_url_list = self.main_data['Photo_URL'].to_list()
                logger.info(f'ProcessReport | read_report_data | Report was read successful')
            return self.main_data, self.photo_url_list

        except Exception as err:
            logger.error(f'ProcessReport | read_report_data | {err.__class__.__name__} {err}')

    def process_result_data(self):
        self.read_report_data()
        if self.photo_url_list is not None and len(self.photo_url_list) > 0:
            self.detection_data = ModelInference(self.package_path, self.photo_url_list).inference_image_process()
            try:
                # Update Results in Result file - checkpoint
                self.main_data['Photo_URLx'] = self.main_data['Photo_URL'].map(str)
                self.main_data.set_index('Photo_URLx', inplace=True)
                self.main_data.update(self.detection_data.set_index('filename'))
                self.main_data.reset_index(inplace=True)
                self.main_data = self.main_data.drop(['Photo_URLx'], axis=1, errors='ignore')

                # Format column data types
                for i in self.main_data.columns[18:]:
                    self.main_data[i] = self.main_data[i].apply(np.float64)

                if self.role == 'TT':
                    # Calculate Avg_SKU_per_shelf_level
                    self.main_data['Sum'] = (self.main_data[['Lays_Total', 'EP_Total', 'Lays_Max_Total']]
                                             .apply(pd.to_numeric, downcast='integer', errors='coerce')
                                             .sum(1, skipna=False))

                    self.main_data['Avg_SKU_per_shelf_level'] = (
                        (self.main_data['Sum'] / self.main_data['Shelf_number'])
                        .replace(np.inf, 0))

                    self.main_data['Avg_SKU_per_shelf_level'] = (self.main_data['Avg_SKU_per_shelf_level']
                                                                 .apply(pd.to_numeric, downcast='float',
                                                                        errors='coerce')
                                                                 .round(1))
                    # Avg_SKU_per_shelf_level calculation
                    self.main_data['Audit_result'] = np.where((self.main_data['Avg_SKU_per_shelf_level']
                                                               >= self.main_data['SKU_Target']), 'Y', 'N')

                    # Drop un-used col
                    self.main_data = self.main_data.drop(['Sum'], axis=1, errors='ignore')

                if self.main_data is not None:
                    pack_report_file(self.main_data, self.arc_file_path, self.xlsx_file_name)
                    logger.info(f'ProcessReport | process_result_data | Result is completed')
                    self.clean_workplace()

            except Exception as err:
                logger.error(f'ProcessReport | process_result_data | {err.__class__.__name__} {err}')
                self.clean_workplace()
        else:
            logger.warning(f'ProcessReport | process_result_data | Photo URL is empty')
            self.clean_workplace()

    def clean_workplace(self):
        if self.package_path.exists():
            shutil.rmtree(self.package_path)
            logger.info(f'ProcessReport| Package folder was wiped')
