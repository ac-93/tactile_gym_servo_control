from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy
import cv2
from scipy import ndimage
from skimage.util import random_noise


def convert_image_uint8(image):
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = 255 * image  # Now scale by 255
    return image.astype(np.uint8)


def pixel_diff_norm(frames):
    ''' Computes the mean pixel difference between the first frame and the
        remaining frames in a Numpy array of frames.
    '''
    n, h, w, c = frames.shape
    pdn = [cv2.norm(frames[i], frames[0], cv2.NORM_L1) / (h * w)
           for i in range(1, n)]
    return np.array(pdn)


def load_video_frames(filename):
    ''' Loads frames from specified video 'filename' and returns them as a
        Numpy array.
    '''
    frames = []
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        captured, frame = vc.read()
        if captured:
            frames.append(frame)
        while captured:
            captured, frame = vc.read()
            if captured:
                frames.append(frame)
        vc.release()
    return np.array(frames)


def process_image(
    image,
    gray=True,
    bbox=None,
    dims=None,
    stdiz=False,
    normlz=False,
    thresh=False,
    circle_mask_radius=None,
):
    ''' Process raw image (e.g., before applying to neural network).
    '''
    if gray:
        # Convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Add channel axis
        image = image[..., np.newaxis]

    if bbox is not None:
        # Crop to specified bounding box
        x0, y0, x1, y1 = bbox
        image = image[y0:y1, x0:x1]

    if circle_mask_radius is not None:
        image = apply_circle_mask(image, circle_mask_radius)

    if dims is not None:
        if isinstance(dims, list):
            dims = tuple(dims)

        # Resize to specified dims
        image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)

        # Add channel axis
        image = image[..., np.newaxis]

    if thresh:
        # Use adaptive thresholding to create binary image
        image = threshold_image(image)
        image = image[..., np.newaxis]

    if stdiz:
        # Convert to float and standardise on a per frame basis
        # position of this is important
        image = per_image_standardisation(image.astype(np.float32))

    if normlz:
        # Convert to float and standardise on a per frame basis
        # position of this is important
        image = image.astype(np.float32) / 255.0

    return image


def augment_image(
    image,
    rshift=None,
    rzoom=None,
    brightlims=None,
    noise_var=None
):
    ''' Process raw image (e.g., before applying to neural network).
    '''

    if rshift is not None:
        # Apply random shift to image
        wrg, hrg = rshift
        image = random_shift_image(image, wrg, hrg)

    if rzoom is not None:
        # Apply random zoom to image
        image = random_zoom_image(image, rzoom)

    if brightlims is not None:
        # Add random brightness/contrast variation to the image
        image = random_image_brightness(image, brightlims)

    if noise_var is not None:
        # Add random noise to the image
        image = random_image_noise(image, noise_var)

    return image


def threshold_image(image):
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)
    return image


def apply_circle_mask(image, radius=110):
    hh, ww = image.shape[:2]
    hc = hh // 2
    wc = ww // 2

    mask = np.ones(shape=(hh, ww))
    mask = cv2.circle(
        mask,
        (wc, hc),
        radius, 0, -1
    )

    image[mask == 1] = 0
    return image


def random_image_brightness(image, brightlims):

    if image.dtype != np.uint8:
        raise ValueError(
            'This random brightness should only be applied to uint8 images on a 0-255 scale')

    a1, a2, b1, b2 = brightlims
    alpha = np.random.uniform(a1, a2)  # Simple contrast control
    beta = np.random.randint(b1, b2)  # Simple brightness control
    new_image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)

    return new_image


def random_image_noise(image, noise_var):
    new_image = (random_noise(image,  var=noise_var) * 255).astype(np.uint8)
    return new_image


def per_image_standardisation(image):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.sqrt(((image - mean)**2).mean(axis=(0, 1), keepdims=True))
    t_image = (image - mean) / (std+1e-6)
    return t_image


def random_shift_image(x, wrg, hrg, fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    """
    h, w = x.shape[0], x.shape[1]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, fill_mode=fill_mode, cval=cval)
    return x


def random_zoom_image(x, zoom_range, fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    x = apply_affine_transform(x, zx=zx, zy=zy, fill_mode=fill_mode, cval=cval)
    return x


def apply_affine_transform(x, theta=0, tx=0, ty=0, zx=1, zy=1,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[0], x.shape[1]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, 2, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, 3)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
