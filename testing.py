from logger_class import Logger
from process_report_class import ProcessReport
from inference_class import InferenceModel
from pathlib import Path
from warnings import simplefilter
from time import perf_counter

logger = Logger().get_logger('testing')
logger.info('Logger from testing.py')

root_dir = Path().resolve()
simplefilter("ignore")


class MainProcess:
    def __init__(self):
        self.start = perf_counter()
        self.report_path = Path(root_dir, "Reports/")
        self.result_path = Path(root_dir, "Results/")
        self.num_of_report = 0
        self.select_report = None

    def process_select_report(self, debug=False):
        report_list = [Path(x).name for x in self.report_path.iterdir() if Path(x).name.endswith(".xlsx")]

        for file in range(len(report_list)):
            self.num_of_report += 1
            # print(f'INFO: {file}. {report_list[file]}')

        if self.num_of_report > 0:
            # user_input = int(input(f'Please select the report to process: '))
            # self.select_report = Path(self.report_path, str(report_list[user_input]))
            self.select_report = Path(self.report_path, str(report_list[2]))
            report_process = ProcessReport(self.select_report)
            report_process.get_result_template()
            url_list = report_process.get_url_listing()
            report_process.save_result_file(save=True)
            inference_process = InferenceModel()
            detection_df = inference_process.inference_process(url_list, debug=debug)
            report_process.append_detection_data(detection_df, debug=debug)
        else:
            print(f'INFO: Report folder was empty.')


if __name__ == '__main__':
    MainProcess().process_select_report(debug=True)

