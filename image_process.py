import math
import cv2
import numpy as np
# from scipy.signal import convolve2d
from config_parser import config_parser

np.seterr(divide='ignore', invalid='ignore')

image_sharpness_threshold = None


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


def image_sharpness(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """local_perceived_sharpness"""
    laplacian_sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_sharpness


# def estimate_noise(image):
#     H, W = image.shape[:2]
#     M = [[1, -2, 1],
#          [-2, 4, -2],
#          [1, -2, 1]]
#     sigma = np.sum(np.sum(np.absolute(convolve2d(image, M))))
#     sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
#     return sigma


def laplacian_variance(img_path):  # Work perfect

    # Convert to Grayscale
    im_gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

    # Remove Noise
    im_denoise = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # laplacian extraction
    im_laplacian_64 = cv2.Laplacian(im_denoise, cv2.CV_64F)
    fm = cv2.Laplacian(im_gray, cv2.CV_64F).var()

    # Extract variances
    mean, stddev = cv2.meanStdDev(im_laplacian_64)

    return float(fm), float(stddev[0])


def brightness_variance(img):
    # Crop Image
    y = 250
    x = 250
    h = 640
    w = 640
    crop_img = img[y:y + h, x:x + w]

    # Resize image to 50x50
    dim = 50
    image = cv2.resize(crop_img, (dim, dim))

    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)

    return float(np.mean(L))


def lanscape_check(img):
    (h, w) = img.shape[:2]
    if h < w:
        return True
    else:
        return False


def rotate90_img(img):
    (h, w) = img.shape[:2]
    if h < w:
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by -90 degrees around the image
        M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        img_rotated = cv2.warpAffine(img, M, (w, h))
        return img_rotated
    else:
        return img


def image_processing(img):
    global image_sharpness_threshold
    image_sharpness_threshold = config_parser(section='config', mode='read', field='image_sharpness')

    sliced_img = image_slicing(img)
    img_sharpness = [image_sharpness(x) for x in sliced_img]
    lap_mean = np.mean(img_sharpness)
    if lap_mean <= float(image_sharpness_threshold):
        return "Bad quality"
    else:
        return "Accepted quality"

# if __name__ == '__main__':
#     folder = r'C:\Users\80230470\Desktop\New folder (2)'
# f_list = [Path(x) for x in Path.iterdir(Path(folder)) if x.name.endswith('.jpg')]
# arr = []
# for i in f_list:
#     img_read = cv2.imread(str(i))
#     var_lap, dn_mean_lap = laplacian_variance(img_read)
#     arr.append([i.name, var_lap, dn_mean_lap])
#
# df = pd.DataFrame(arr, columns=['filename', 'var_lap', 'dn_mean_lap'])
# df.to_excel('blur_output_2.xlsx', index=False)
