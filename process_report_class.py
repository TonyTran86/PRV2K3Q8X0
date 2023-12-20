from logger_class import Logger

import numpy as np
import pandas as pd

from warnings import simplefilter
from pathlib import Path
from thread_executor_class import ThreadExecutor
from time import perf_counter

logger = Logger().get_logger('ProcessReport_class')
logger.info('Logger from process_report_class.py')
root_dir = Path().resolve()
simplefilter("ignore")


class ProcessReport:
    def __init__(self, report_path):
        self.start = perf_counter()
        self.f_path = Path(report_path)
        self.f_root = Path(report_path).parent
        self.f_full_name = self.f_path.name
        self.f_name = self.f_path.name
        self.f_extension = self.f_path.suffix
        self.result_file_path = None

        self.df_read = pd.DataFrame()
        self.df_default = pd.DataFrame()
        self.df_template = pd.DataFrame()
        self.df_comp_list = pd.DataFrame()

        self.result_path = Path(root_dir, 'Results/')
        self.temp_path = Path(self.result_path, 'Temp/')
        self.split_path = Path(root_dir, 'Reports/Split_reports/')
        self.split_folder_path = None
        self.split_folder_name = None

        self.template = None
        self.new_name = None
        self.chunk = 0
        self.length = 0
        self.list_zip = []

    def get_dataframe(self):
        try:
            self.df_read = pd.read_excel(self.f_path, engine="openpyxl", index_col=False, dtype="unicode",
                                         na_values=['NA'])
            # ProcessReport.print_out(f'INFO: ProcessReport | get_dataframe: report file was read')
            return self.df_read

        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | get_dataframe: {err.__class__.__name__} {err}', mode='error')

    def report_template_checking(self, debug=False):
        try:
            if self.df_read.empty:
                self.get_dataframe()

            col_count = len(self.df_read.columns)
            cols = ['Year', 'Month', 'Answered Date', 'Store Region', 'Store State', 'User ID', 'Merchandiser  Name',
                    'Store Format', 'Storecode', 'Store Name', 'Time', 'Sync Date', 'Task', 'Category', 'Sub Category',
                    'Product Barcode', 'Description', 'Response/ Snap', 'Response/ Snap.1', 'Map URL']

            h_chk = self.df_read.columns.isin(cols).all()

            if debug:
                ProcessReport.print_out(f'DEBUG: ProcessReport | col_count: {col_count}, col_check: {h_chk}')

            if col_count == 20 and h_chk:
                return True
            else:
                return False

        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | get_template_checking: {err.__class__.__name__} {err}', mode='error')

    def result_template_checking(self):
        try:
            if self.df_read.empty:
                self.get_dataframe()
            data = pd.read_excel(self.df_read, engine="openpyxl", index_col=False, dtype="unicode", na_values=['NA'])
            if pd.Series(
                    ['Year', 'Month', 'Store_Region', 'Store_State', 'Merchandiser_Name', 'Store_Format', 'Storecode',
                     'Store_Name', 'Sync_Date', 'Task', 'Category', 'Photo_URL', 'Status']).isin(data.columns).all():
                return True
            else:
                return False
        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | result_template_checking: {err.__class__.__name__} {err}', mode='error')

    def get_result_template(self):
        self.template = self.report_template_checking()
        try:
            if self.template:
                # Remove empty & non-data cols
                self.df_default = self.df_read.dropna(how='all')
                self.df_default = self.df_read[self.df_read['Category'].str.contains("CHỤP ẢNH")]

                # Rename col headers: replace space in column name
                self.df_default.columns = [c.replace(' ', '_') for c in self.df_default.columns]

                # Extract Date from HH_Submit_Date as Submit_Date col
                self.df_default['Sync_Date'] = pd.to_datetime(self.df_default['Sync_Date'], errors='coerce')
                self.df_default = self.df_default.rename(columns={'Response/_Snap.1': 'Photo_URL'})

                # Modify dataframe
                self.df_default['Store_Name_2'] = self.df_default['Store_Name'].str[8:50]
                self.df_default['Status'] = ''

                col_names = ['Lays_Total', 'Lays_10M', 'Lays_20M', 'Lays_6M', 'Lays_Max', 'Lays_Stax', 'EP_Total',
                             'EP_10M',
                             'EP_20M', 'Doritos', 'Nutz']
                default_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                for col_name, default_value in zip(col_names, default_values):
                    self.df_default[col_name] = default_value

                # Drop un-used cols
                self.df_default = self.df_default.drop(
                    ['Answered_Date', 'Time', 'Sub_Category', 'Product_Barcode', 'Response/_Snap',
                     'Map_URL', 'Store_Name', 'Description'], axis=1, errors='ignore')

                self.df_default = self.df_default.rename(columns={'Store_Name_2': 'Store_Name'})
                self.df_default = self.df_default.rename(columns={'Merchandiser__Name': 'Merchandiser_Name'})

                # Re-order
                cols = ['Year', 'Month', 'Store_Region', 'Store_State', 'Merchandiser_Name', 'Store_Format',
                        'Storecode',
                        'Store_Name', 'Sync_Date', 'Task', 'Category', 'Photo_URL', 'Status', 'Lays_Total', 'Lays_6M',
                        'Lays_10M', 'Lays_20M', 'Lays_Max', 'Lays_Stax', 'EP_Total', 'EP_10M', 'EP_20M', 'Doritos',
                        'Nutz']

                self.df_default = self.df_default.reindex(columns=cols)
                ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} '
                                        f'| ProcessReport | get_result_template: Result template was created')
            else:
                self.df_default = self.df_read
                ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} '
                                        f'| ProcessReport | get_result_template: Report is default template')
        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | get_result_template: '
                                    f'{err.__class__.__name__} {err}', mode='error')

        return self.df_default

    def get_url_listing(self):
        try:
            self.list_zip = self.df_default['Photo_URL']
            ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} | ProcessReport '
                                    f'| get_url_listing: URL listing was extracted')
            return self.list_zip

        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | get_url_listing: '
                                    f'{err.__class__.__name__} {err}', mode='error')

    def get_split_rows(self, arg):
        from_row, to_row = arg[0], arg[1]
        data = self.df_default.iloc[from_row:to_row]
        filename = str(from_row) + '_' + str(to_row) + '_' + '.xlsx'
        save_split_file = Path(self.split_folder_path, self.split_folder_name + '_' + filename)
        data.to_excel(save_split_file, index=False)
        ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} | '
                                f'ProcessReport | get_split_rows | Report {filename} : saved')

    # @staticmethod
    def get_split_rows_list(self, arg):
        data_range = len(arg)

        if data_range < 1000:
            chunk_range = 500
        else:
            chunk_range = 1000

        total_even_step = int(data_range / chunk_range)
        even_range = int(data_range / chunk_range) * chunk_range
        num, remain_range = divmod(data_range, chunk_range)

        ProcessReport.print_out(f'INFO: ProcessReport | get_split_rows_list: Data length: {data_range}')
        ProcessReport.print_out(f'INFO: ProcessReport | get_split_rows_list: Total even steps: {total_even_step}')
        ProcessReport.print_out(f'INFO: ProcessReport | get_split_rows_list: Even range: {even_range}')
        ProcessReport.print_out(f'INFO: ProcessReport | get_split_rows_list: Remain: {remain_range}')

        if remain_range > 0:
            last_step = total_even_step + 1
        else:
            last_step = total_even_step

        x_range = [x for x in range(0, data_range, chunk_range)]
        x_range.append(even_range + remain_range)

        from_row = []
        to_row = []

        for i in range(last_step):
            from_row.append(x_range[i])
            to_row.append(x_range[i + 1])

        list_zip = zip(from_row, to_row)
        ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} '
                                f'| ProcessReport | get_split_rows_list: from_to_row was extracted')

        return list_zip

    def save_split_row_files(self):  # save each split result file
        self.split_folder_name = Path(self.f_path).stem
        self.split_folder_path = self.split_path
        self.df_comp_list = self.df_default['Photo_URL']

        # Process save split row files
        com_list = self.get_split_rows_list(self.df_comp_list)  # Calculate from row, to row
        ThreadExecutor.cf_ThreadPoolExecutor_submit_no_result(self.get_split_rows, com_list)
        ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} | '
                                f'ProcessReport | save_split_row_files: split files were saved')

    def save_result_file(self, file_name_suffix='_result.xlsx', save=True):  # Save result by route
        # save_file_name = Path(self.temp_path, str(arg) + '_duplicate_result.xlsx')
        # df = self.df_default[self.df_default['Route_Code'] == arg]
        # df.to_excel(save_file_name, index=False)

        self.result_file_path = Path(self.result_path, Path(self.f_path).stem + file_name_suffix)
        if save:
            if not self.result_path.exists():
                Path.mkdir(self.result_path, exist_ok=True)
            self.df_default.to_excel(self.result_file_path, index=False, sheet_name='Export')
            ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} | '
                                    f'ProcessReport | save_result_file: Result file was exported')
        else:
            return self.result_file_path

    def append_detection_data(self, detection_data, debug=False):
        try:
            # Add Status column with conditional, rename column, drop unused column
            detection_data['Status'] = np.where((detection_data['url_status'] == 'error') |
                                                (detection_data['a_status'] == 'error') |
                                                (detection_data['b_status'] == 'error'), 'error', 'success')

            col_name = ['poca', 'lays', 'laysmax', 'doritos', 'stax', 'lays10', 'nutz']
            new_name = ['EP_Total', 'Lays_Total', 'Lays_Max', 'Doritos', 'Lays_Stax', 'Nutz', 'Lays_10M']
            drop_name = ['url_status', 'a_status', 'b_status']

            for i in range(len(col_name)):
                if col_name[i] in detection_data.columns:
                    detection_data.rename(columns={col_name[i]: new_name[i]}, inplace=True)

            detection_data = detection_data.drop(drop_name, axis=1, errors='ignore')

            # Assign to new dataframe
            df = detection_data

            if debug:
                df.to_excel('detection_transformed_data.xlsx', index=False)
                ProcessReport.print_out(f'DEBUG: {perf_counter()-self.start:.2f} | ProcessReport | '
                                        f'ProcessReport | detection_transformed_data.xlsx was saved')

            if self.df_default.empty:
                # Read result file
                self.df_default = pd.read_excel(self.result_file_path, engine="openpyxl", index_col=False,
                                                dtype="unicode", na_values=['NA'])

            # Update Results in Result file - checkpoint
            self.df_default['Photo_URLx'] = self.df_default['Photo_URL'].map(str)
            self.df_default.set_index('Photo_URLx', inplace=True)
            self.df_default.update(df.set_index('filename'))
            self.df_default.reset_index(inplace=True)  # to recover the initial structure

            # Drop un-used col
            self.df_default = self.df_default.drop(['Photo_URLx'], axis=1, errors='ignore')

            '''Debug'''
            # result_data.to_excel('data_output_03.xlsx', index=False)
            main_col = ['Lays_Total', 'Lays_6M', 'Lays_10M', 'Lays_20M', 'Lays_Max', 'Lays_Stax', 'EP_Total', 'EP_10M',
                        'EP_20M', 'Doritos', 'Nutz']

            for i in self.df_default.columns:
                if i in main_col:
                    self.df_default[i] = self.df_default[i].apply(np.float64)

            self.df_default.to_excel(self.result_file_path, index=False)
            ProcessReport.print_out(f'INFO: {perf_counter()-self.start:.2f} | '
                                    f'ProcessReport | Detection result appending was completed')

        except Exception as err:
            ProcessReport.print_out(f'ERROR: ProcessReport | append_detection_data: '
                                    f'{err.__class__.__name__} {err}', mode='error')

    # def merge_detection_files():  # for combination
    #     current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    #     name_string = 'Combination_' + str(current_datetime) + '_' + '.xlsx'
    #
    #     xlsx_files = [y for y in Path(combine_path).iterdir() if y.name.endswith('.xlsx')]
    #     merged = pd.concat(map(pd.read_excel, xlsx_files), axis=0)
    #     save_name = Path(result_path, name_string)
    #     merged.to_excel(save_name, index=False)

    @staticmethod
    def print_out(input_string, mode='debug'):  # debug, error, info
        print(input_string)
        if mode == 'error':
            logger.error(input_string)
        else:
            logger.debug(input_string)


# if __name__ == '__main__':
    # file = r'C:\Users\80230470\PycharmProjects\Test_build_app\OT_app\Reports\Report_Sample_02.xlsx'
    # main_process = ProcessReport(file)
    # # main_df = main_process.get_dataframe()
    # main_df = main_process.get_result_template()
    # url_list = main_process.get_url_listing()
    # main_process.save_result_file(save=True)
