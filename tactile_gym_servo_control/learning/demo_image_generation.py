"""
python demo_image_generation.py -t surface_3d
python demo_image_generation.py -t edge_2d
python demo_image_generation.py -t edge_3d
python demo_image_generation.py -t edge_5d
python demo_image_generation.py -t surface_3d edge_2d edge_3d edge_5d
"""


import os
import argparse
import cv2
import numpy as np
import torch

from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator
from tactile_gym_servo_control.learning.image_generator import numpy_collate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = os.path.join(os.path.dirname(__file__), '../data')


def run(data_dirs, learning_params, image_processing_params, augmentation_param):

    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = ImageDataGenerator(data_dirs=data_dirs, **generator_args)

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for (i_batch, sample_batched) in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        images = sample_batched['images']
        labels = sample_batched['labels']
        cv2.namedWindow("example_images")

        for i in range(learning_params['batch_size']):
            for key, item in labels.items():
                print(key.split('_')[0], ': ', item[i])
            print('')

            # convert image to opencv format, not pytorch
            image = np.moveaxis(images[i], 0, -1)
            cv2.imshow("example_images", image)
            k = cv2.waitKey(500)
            if k == 27:    # Esc key to stop
                exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
    )
    args = parser.parse_args()
    tasks = args.tasks

    learning_params = {
        'batch_size':  8,
        'shuffle': True,
        'n_cpu': 1,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': False,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom':   None,
        'brightlims': None,
        'noise_var': None,
    }

    data_dirs = [
        os.path.join(data_path, task, 'tap', 'train') for task in tasks
        # os.path.join(data_path, task, 'tap', 'val') for task in tasks
    ]

    run(data_dirs, learning_params, image_processing_params, augmentation_params)
