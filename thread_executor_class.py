import pandas as pd

from warnings import simplefilter
from pathlib import Path
from math import ceil
from datetime import datetime
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from logger_class import Logger

logger = Logger().get_logger('ThreadExecutor_class')
logger.info('Logger from thread_executor_class.py')

root_dir = Path().resolve()
simplefilter("ignore")


class ThreadExecutor(object):

    @staticmethod
    def multi_threading(func, args):
        pros_num = cpu_count() - 1
        size = 3
        output = pd.DataFrame()
        date = datetime.now()
        ThreadExecutor.print_out(f'INFO: Processors: {pros_num} | Chunks: {size} |'
                                 f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            pool = ThreadPool(pros_num)
            for f in pool.map(func, args, chunksize=size):
                df = pd.DataFrame([f], columns=f.keys())
                output = pd.concat([output, df], ignore_index=True)
            pool.close()
            ThreadExecutor.print_out(f'Process finished in: {time() - t1}')
            return output

        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def multithreading_ThreadPool_map(func, iterable):
        processors = cpu_count() - 1
        task_chunk = 3
        date = datetime.now()
        output = pd.DataFrame()
        ThreadExecutor.print_out(f'INFO: Processors: {processors} | Chunks: {task_chunk}'
                                 f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPool(processors) as executor:
                for result in executor.map(func, iterable, chunksize=task_chunk):
                    df = pd.DataFrame([result], columns=result.keys())
                    output = pd.concat([output, df], ignore_index=True)
                ThreadExecutor.print_out(f'Process finished in: {time() - t1}')
                return output
        except Exception as err:
            ThreadExecutor.print_out(f'multithreading_ThreadPool error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def cf_ThreadPoolExecutor_map(func, iterable):
        iter_length = len(iterable)
        processors = cpu_count() - 1
        if iter_length < 100:
            task_chunk = iter_length
        else:
            task_chunk = ceil((iter_length * 2) / 100)
        date = datetime.now()
        output = pd.DataFrame()
        ThreadExecutor.print_out(
            f'INFO: Processors: {processors} | Chunks: {task_chunk} | Data length: {iter_length} | '
            f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPoolExecutor(max_workers=processors) as executor:
                for result in executor.map(func, iterable, chunksize=iter_length):
                    df = pd.DataFrame([result], columns=result.keys())
                    output = pd.concat([output, df], ignore_index=True)
                print(f'Process finished in: {time() - t1}')
                return output

        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def cf_ThreadPoolExecutor_submit(func, iterable):  # submit method is better but unordered results
        # iter_length = len(iterable)
        processors = cpu_count() - 1
        task_chunk = 5
        date = datetime.now()
        output = pd.DataFrame()
        ThreadExecutor.print_out(f'INFO: Processors: {processors}/{cpu_count()} | Chunks: {task_chunk} |'
                                 f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPoolExecutor(max_workers=processors) as executor:
                futures = [executor.submit(func, i) for i in iterable]
                for f in as_completed(futures):
                    # for f in futures:  # unordered results
                    # if f.result():
                    # df = pd.DataFrame(f.result(), columns=f.result().keys())
                    df = pd.DataFrame([f.result()])
                    # print(f.result())
                    output = pd.concat([output, df], ignore_index=True)
                # output.to_excel('test_output.xlsx', index=False)
                ThreadExecutor.print_out(f'Process finished in: {time() - t1}')
                return output
        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def cf_ThreadPoolExecutor_submit_no_result(func, iterable):  # submit method is better but unordered results
        processors = cpu_count() - 1
        task_chunk = 5
        date = datetime.now()
        ThreadExecutor.print_out(f'INFO: Processors: {processors} | Chunks: {task_chunk} |'
                                 f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPoolExecutor(max_workers=processors) as executor:
                futures = [executor.submit(func, i) for i in iterable]
                for _ in as_completed(futures):
                    pass
                ThreadExecutor.print_out('Process finished in: {}'.format(time() - t1))

        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def cf_ThreadPoolExecutor_submit_universal(func, iterable):  # submit method is better but unordered results
        # iter_length = len(iterable)
        processors = cpu_count() - 1
        task_chunk = 5
        date = datetime.now()
        output = pd.DataFrame()
        ThreadExecutor.print_out(f'INFO: Processors: {processors} | Chunks: {task_chunk} | '
                                 f'Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPoolExecutor(max_workers=processors) as executor:
                futures = [executor.submit(func, i) for i in iterable]
                for f in as_completed(futures):
                    df = pd.DataFrame(f.result())
                    output = pd.concat([output, df], ignore_index=True)
                ThreadExecutor.print_out(f'Process finished in: {time() - t1}')
                return output
        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def cf_ThreadPoolExecutor_submit_save_label(func, iterable):  # submit method is better but unordered results
        processors = cpu_count() - 1
        task_chunk = 3
        date = datetime.now()
        # output = pd.DataFrame()
        ThreadExecutor.print_out(f'INFO: Processors: {processors} | Chunks: {task_chunk} | Start: {date.strftime("%H:%M:%S.%f")}')
        try:
            t1 = time()
            with ThreadPoolExecutor(max_workers=processors) as executor:
                futures = [executor.submit(func, i) for i in iterable]
                for _ in as_completed(futures):
                    pass
                    # print('Done')
                    # df = pd.DataFrame(f.result())
                    # output = pd.concat([output, df], ignore_index=True)
                ThreadExecutor.print_out(f'Process finished in: {time() - t1}')
                # return output

        except Exception as err:
            ThreadExecutor.print_out(f'cf_ThreadPoolExecutor error: {err.__class__.__name__} {err}', mode='error')

    @staticmethod
    def print_out(input_string, mode='debug'):  # debug, error, info
        print(input_string)
        if mode == 'error':
            logger.error(input_string)
        else:
            logger.debug(input_string)

    # def tqdm_parallel_map(fn, *iterables):
    #     """ use tqdm to show progress"""
    #     executor = concurrent.futures.ProcessPoolExecutor()
    #     futures_list = []
    #     for iterable in iterables:
    #         futures_list += [executor.submit(fn, i) for i in iterable]
    #     for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list)):
    #         yield f.result()
    #
    # def multi_cpu_dispatcher_process_tqdm(data_list, single_job_fn):
    #     """ multi cpu dispatcher """
    #     output = pd.DataFrame()
    #     for result in tqdm_parallel_map(single_job_fn, data_list):
    #         df = pd.DataFrame([result], columns=result.keys())
    #         output = pd.concat([output, df], ignore_index=True)
    #     return output
