"""
python test_cnn.py -t surface_3d
python test_cnn.py -t edge_2d
python test_cnn.py -t edge_3d
python test_cnn.py -t edge_5d
python test_cnn.py -t surface_3d edge_2d edge_3d edge_5d
"""
import os
import argparse
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.learning_utils import acc_metric
from tactile_gym_servo_control.learning.learning_utils import err_metric
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator
from tactile_gym_servo_control.learning.plot_tools import plot_error


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = os.path.join(os.path.dirname(__file__), '../data/')
model_path = os.path.join(os.path.dirname(__file__), '../learned_models')

# tolerances for accuracy metric
POS_TOL = 0.25  # mm
ROT_TOL = 1.0  # deg


def test_cnn(
    task,
    model,
    label_names,
    pose_limits,
    learning_params,
    image_processing_params,
    save_dir_name,
    device='cpu'
):

    # data dir
    # can specifiy multiple directories that get combined in generator
    test_data_dirs = [
        os.path.join(data_path, task, 'tap', 'val')
    ]

    # set generators and loaders
    test_generator = ImageDataGenerator(data_dirs=test_data_dirs, **image_processing_params)

    test_loader = torch.utils.data.DataLoader(
        test_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    pred_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
    targ_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

    # complete dateframe of accuracy and errors
    acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
    err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

    for i, batch in enumerate(test_loader):

        # get inputs
        inputs, labels_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # count correct for accuracy metric
        predictions_dict = decode_pose(outputs, label_names, pose_limits)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
        batch_targ_df = pd.DataFrame.from_dict(labels_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

        # get errors and accuracy
        batch_err_df = err_metric(labels_dict, predictions_dict, label_names)
        batch_acc_df = acc_metric(batch_err_df, label_names, POS_TOL, ROT_TOL)

        # append error to dataframe
        err_df = pd.concat([err_df, batch_err_df])
        acc_df = pd.concat([acc_df, batch_acc_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    acc_df = acc_df.reset_index(drop=True).fillna(0.0)
    err_df = err_df.reset_index(drop=True).fillna(0.0)

    print("Test Metrics")
    print("test_acc:")
    print(acc_df[[*label_names, 'overall_acc']].mean())
    print("test_err:")
    print(err_df[label_names].mean())

    # plot full error graph
    plot_error(
        pred_df,
        targ_df,
        err_df,
        label_names,
        pose_limits,
        save_file=os.path.join(save_dir_name, 'error_plot.png'),
        show_plot=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cpu'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    device = args.device

    # test the trained networks
    for task in tasks:

        save_dir_name = os.path.join(
            model_path,
            task,
            'tap',
        )

        model_params = load_json_obj(os.path.join(save_dir_name, 'model_params'))
        learning_params = load_json_obj(os.path.join(save_dir_name, 'learning_params'))
        image_processing_params = load_json_obj(os.path.join(save_dir_name, 'image_processing_params'))

        # set the correct accuracy metric and label generator
        out_dim, label_names = import_task(task)

        # get the pose limits used for encoding/decoding pose/predictions
        pose_limits_dict = load_json_obj(os.path.join(save_dir_name, 'pose_limits'))
        pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]

        # create the model
        model = create_model(
            image_processing_params['dims'],
            out_dim,
            model_params,
            saved_model_dir=save_dir_name,
            device=device
        )
        model.eval()

        test_cnn(
            task,
            model,
            label_names,
            pose_limits,
            learning_params,
            image_processing_params,
            save_dir_name,
            device
        )
