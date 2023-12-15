import sys

sys.path.insert(0, 'utils')

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset

# from Photo_Processing.Class_download import read_image_url
# from Photo_Processing.Class_executor import create_executor

root_dir = Path().resolve()
model_path = Path(root_dir, "app_sys/yolov5-master/trained_model/pretrained_model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = torch.load(model_path)
trained_model = trained_model['model_state']


def run_testing_on_dataset(trained_model, dataset_dir, GT_blurry):
    correct_prediction_count = 0
    img_list = os.listdir(dataset_dir)
    for ind, image_name in enumerate(img_list):
        print("Blurry Image Prediction: %d / %d images processed.." % (ind, len(img_list)))

        # Read the image
        img = cv2.imread(os.path.join(dataset_dir, image_name), 0)

        prediction = is_image_blurry(trained_model, img, threshold=0.5)

        if prediction == GT_blurry:
            correct_prediction_count += 1
    accuracy = correct_prediction_count / len(img_list)
    return accuracy


# def is_image_blurry(img):
#     feature_extractor = feature_extractor_module.featureExtractor()
#     accumulator = []
#
#     # Resize the image by the down sampling factor
#     feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])
#
#     # compute the image ROI using local entropy filter
#     feature_extractor.compute_roi()
#
#     # extract the blur features using DCT transform coefficients
#     extracted_features = feature_extractor.extract_feature()
#     extracted_features = np.array(extracted_features)
#
#     if len(extracted_features) == 0:
#         return True
#     test_data_loader = DataLoader(dataloader_module.TestDataset(extracted_features), batch_size=1, shuffle=False)
#
#     # trained_model.test()
#     for batch_num, input_data in enumerate(test_data_loader):
#         x = input_data
#         x = x.to(device).float()
#
#         output = trained_model(x)
#         _, predicted_label = torch.max(output, 1)
#         accumulator.append(predicted_label.item())
#
#     val = np.mean(accumulator)
#     # prediction = np.mean(accumulator) < threshold
#     return val

# threshold=0.488

def is_image_blurry(img, threshold):
    feature_extractor = featureExtractor()
    accumulator = []

    # Resize the image by the down sampling factor
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if len(extracted_features) == 0:
        return True
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    # trained_model.test()
    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    val = np.mean(accumulator)
    if val <= threshold:
        return 'Bad quality'
    else:
        return 'Accepted quality'
    # prediction = np.mean(accumulator) < threshold
    # return val

# def get_url_list(arg):
#     read_data = pd.read_excel(arg, sheet_name='Export', engine="openpyxl", index_col=False, dtype="unicode",
#                               na_values=['NA'])
#     subset_rows = read_data.iloc[1:1000]
#     url_list = subset_rows['Photo_URL']
#     return url_list
#
#
# def predict_image_file(arg):
#     output = {'img_name': [], 'prediction': []}
#     output['img_name'].append(os.path.basename(arg))
#     try:
#         img_gray = cv2.imread(arg, 0)
#         prediction_val = is_image_blurry(img_gray)
#         if prediction_val < 0.488:
#             output['prediction'].append('Blur')
#         else:
#             output['prediction'].append('Good')
#
#         return output
#     except Exception as err:
#         print('predict_image_file error' + str(err.__class__.__name__) + str(err))
#         output['prediction'].append('error')
#
#
# def predict_image_url(arg):
#     output = {'img_url': [], 'img_name': [], 'score': [], 'prediction': []}
#     output['img_url'].append(arg)
#     output['img_name'].append(os.path.basename(arg))
#     try:
#         img_gray = read_image_url(arg, mode=0)
#         if img_gray is not None:
#             score = is_image_blurry(img_gray)
#             output['score'].append(score)
#             if score < 0.4:
#                 output['prediction'].append('Blur')
#             else:
#                 output['prediction'].append('Good')
#         else:
#             output['score'].append(0.0)
#             output['prediction'].append('Invalid')
#
#         return output
#     except Exception as err:
#         print('predict_image_url error' + str(err.__class__.__name__) + str(err))
#         output['score'].append(0.0)
#         output['prediction'].append('error')

# def process_image_folder_prediction(arg, save_name=None):
#     img_list = [os.path.join(arg, x) for x in os.listdir(arg) if x.endswith('.jpg')]
#     data = create_executor.cf_ThreadPoolExecutor_submit(predict_image_file, img_list)
#     save_file = save_name + '.xlsx'
#     data.to_excel(save_file, index=False)
#
#
# def process_image_url_dataframe_prediction(arg, save_name=None):
#     url_list = get_url_list(arg)
#     data = create_executor.cf_ThreadPoolExecutor_submit(predict_image_url, url_list)
#     save_file = save_name + '.xlsx'
#     data.to_excel(save_file, index=False)


# if __name__ == '__main__':
#     NR_report_file = r'C:\Users\80230470\PycharmProjects\Test_build_app\version_1.0.9\build\Workplace_1.2.3\Results' \
#                      r'\NR_2023_05_01_13_result.xlsx'
#
#     ST_report_file = r'C:\Users\80230470\PycharmProjects\Test_build_app\version_1.0.9\build\Workplace_1.2.3\Results' \
#                      r'\ST_2023_05_01_13_result.xlsx'
#
#     url_test = 'https://pepsicodms.com/vnedwrackimage/10000397/3c0r01/at3c0000032320230503094731.jpg'

# process_image_url_dataframe_prediction(NR_report_file, save_name='predict_output_04')

# predict_res = image_prediction(f1)
# print(predict_res)

# dataset_dir = './dataset/defocused_blurred/'
#
# accuracy_blurry_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry=True)
#
# dataset_dir = './dataset/sharp/'
# accuracy_sharp_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry=False)
#
# dataset_dir = './dataset/motion_blurred/'
# accuracy_motion_blur_images = run_testing_on_dataset(trained_model, dataset_dir, GT_blurry=True)
#
# print("========================================")
# print('Test accuracy on blurry forlder = ')
# print(accuracy_blurry_images)
#
# print('Test accuracy on sharp forlder = ')
# print(accuracy_sharp_images)
#
# print('Test accuracy on motion blur forlder = ')
# print(accuracy_motion_blur_images)
