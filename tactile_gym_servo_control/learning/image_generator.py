import numpy as np
import os
import cv2
import pandas as pd
import torch

from tactile_gym_servo_control.utils.image_transforms import process_image
from tactile_gym_servo_control.utils.image_transforms import augment_image


class ImageDataGenerator(torch.utils.data.Dataset):

    def __init__(self,
                 data_dirs,
                 dims=(128, 128),
                 bbox=None,
                 stdiz=False,
                 normlz=False,
                 thresh=None,
                 rshift=None,
                 rzoom=None,
                 brightlims=None,
                 noise_var=None,
                 ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self.dim = dims
        self.bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        # load csv file
        self.label_df = self.load_data_dirs(data_dirs)

    def load_data_dirs(self, data_dirs):

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))
            df['image_dir'] = os.path.join(
                data_dir,
                'images'
            )
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)
        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self.label_df.iloc[index]
        image_filename = os.path.join(row['image_dir'], row['sensor_image'])
        raw_image = cv2.imread(image_filename)

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=True,
            bbox=self.bbox,
            dims=self.dim,
            stdiz=self._stdiz,
            normlz=self._normlz,
            thresh=self._thresh,
        )

        processed_image = augment_image(
            processed_image,
            rshift=self._rshift,
            rzoom=self._rzoom,
            brightlims=self._brightlims,
            noise_var=self._noise_var
        )

        # put the channel into first axis because pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # get label
        target = {
            'x': np.array(row['pose_1']),
            'y': np.array(row['pose_2']),
            'z': np.array(row['pose_3']),
            'Rx': np.array(row['pose_4']),
            'Ry': np.array(row['pose_5']),
            'Rz': np.array(row['pose_6']),
        }

        sample = {'images': processed_image, 'labels': target}

        return sample


def numpy_collate(batch):
    '''
    Batch is list of len: batch_size
    Each element is dict {images: ..., labels: ...}
    Use Collate fn to ensure they are returned as np arrays.
    '''
    # list of arrays -> stacked into array
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    # list of lists/tuples -> recursive on each element
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    # list of dicts -> recursive returned as dict with same keys
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}

    # list of non array element -> list of arrays
    else:
        return np.array(batch)
