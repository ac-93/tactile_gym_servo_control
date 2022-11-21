import os
import numpy as np
import torch

from tactile_gym.utils.general_utils import save_json_obj, load_json_obj

POSE_LABEL_NAMES = ["x", "y", "z", "Rx", "Ry", "Rz"]
POS_LABEL_NAMES = ["x", "y", "z"]
ROT_LABEL_NAMES = ["Rx", "Ry", "Rz"]


def import_task(task_name):
    """
    Returns task specific details.
    """

    if task_name == 'surface_3d':
        out_dim = 5
        label_names = ['z', 'Rx', 'Ry']

    elif task_name == 'edge_2d':
        out_dim = 3
        label_names = ['y', 'Rz']

    elif task_name == 'edge_3d':
        out_dim = 4
        label_names = ['y', 'z', 'Rz']

    elif task_name == 'edge_5d':
        out_dim = 8
        label_names = ['y', 'z', 'Rx', 'Ry', 'Rz']

    else:
        raise ValueError('Incorrect task_name specified')

    return out_dim, label_names


def get_pose_limits(data_dirs, save_dir_name):
    """
     Get limits for poses of data collected, used to encode/decode pose for prediction
     data_dirs is expected to be a list of data directories

     When using more than one data source, limits are taken at the extremes of those used for collection.
    """
    pose_llims, pose_ulims = [], []
    for data_dir in data_dirs:
        collection_params = load_json_obj(os.path.join(data_dir,  'collection_params'))
        pose_llims.append(collection_params['poses_rng'][0])
        pose_ulims.append(collection_params['poses_rng'][1])

    pose_llims = np.min(pose_llims, axis=0)
    pose_ulims = np.max(pose_ulims, axis=0)

    # save limits for use during inference
    pose_limits = {
        'pose_llims': list(pose_llims),
        'pose_ulims': list(pose_ulims),
    }

    save_json_obj(
        pose_limits,
        os.path.join(save_dir_name, 'pose_limits')
    )

    return pose_llims, pose_ulims


def encode_pose(labels_dict, target_label_names, limits, device='cpu'):
    """
    Process raw pose data to NN friendly label for prediction.

    From -> {x, y, z, Rx, Ry, Rz}
    To   -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
    """

    # create tensors for pose limts
    pose_llims = torch.from_numpy(np.array(limits[0])).float().to(device)
    pose_ulims = torch.from_numpy(np.array(limits[1])).float().to(device)

    # encode pose to preictable label
    encoded_pose = []
    for label_name in target_label_names:

        # get the target from the dict
        target = labels_dict[label_name].float().to(device)

        # normalize pose label within limits
        if label_name in POS_LABEL_NAMES:
            llim = pose_llims[POSE_LABEL_NAMES.index(label_name)]
            ulim = pose_ulims[POSE_LABEL_NAMES.index(label_name)]
            norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
            encoded_pose.append(norm_target.unsqueeze(dim=1))

        # sine/cosine encoding of angle
        if label_name in ROT_LABEL_NAMES:
            ang = target * np.pi/180
            ang[ang < 0] += 2 * np.pi
            encoded_pose.append(torch.sin(ang).float().to(device).unsqueeze(dim=1))
            encoded_pose.append(torch.cos(ang).float().to(device).unsqueeze(dim=1))

    # combine targets to make one label tensor
    labels = torch.cat(encoded_pose, 1)

    return labels


def decode_pose(outputs, target_label_names, limits):
    """
    Process NN predictions to raw pose data, always decodes to cpu.

    From  -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
    To    -> {x, y, z, Rx, Ry, Rz}
    """

    # create tensors for pose limts
    pose_llims = torch.from_numpy(np.array(limits[0])).float().cpu()
    pose_ulims = torch.from_numpy(np.array(limits[1])).float().cpu()
    single_element_labels = target_label_names.count("x") + target_label_names.count("y") + target_label_names.count("z")

    # decode preictable label to pose
    decoded_pose = {
        'x': torch.zeros(outputs.shape[0]),
        'y': torch.zeros(outputs.shape[0]),
        'z': torch.zeros(outputs.shape[0]),
        'Rx': torch.zeros(outputs.shape[0]),
        'Ry': torch.zeros(outputs.shape[0]),
        'Rz': torch.zeros(outputs.shape[0]),
    }

    for i, label_name in enumerate(target_label_names):

        if label_name in POS_LABEL_NAMES:
            label_name_idx = i
            predictions = outputs[:, label_name_idx].detach().cpu()

            llim = pose_llims[POSE_LABEL_NAMES.index(label_name)]
            ulim = pose_ulims[POSE_LABEL_NAMES.index(label_name)]
            decoded_predictions = (((predictions + 1) / 2) * (ulim - llim)) + llim
            decoded_pose[label_name] = decoded_predictions

        if label_name in ROT_LABEL_NAMES:
            label_name_idx = single_element_labels + (2 * (i-single_element_labels))
            sin_predictions = outputs[:, label_name_idx].detach().cpu()
            cos_predictions = outputs[:, label_name_idx + 1].detach().cpu()

            pred_rot = torch.atan2(sin_predictions, cos_predictions)
            pred_rot[pred_rot < 0] += 2 * np.pi
            pred_rot = pred_rot * (180.0 / np.pi)

            decoded_pose[label_name] = pred_rot

    return decoded_pose


def acc_metric(labels, predictions, target_label_names, pos_tol=0.2, rot_tol=1.0):
    """
    Accuracy metric for regression problem, counting the number of predictions within a tolerance.

    Position Tolerance (mm), Rotation Tolerance (degrees)
    """

    acc_dict = {}
    total_counted = predictions[target_label_names[0]].shape[0]
    overall_correct = np.ones(total_counted, dtype=bool)
    for label_name in target_label_names:

        if label_name in POS_LABEL_NAMES:
            abs_err = torch.abs(
                labels[label_name] - predictions[label_name]
            ).detach().cpu().numpy()
            correct = (abs_err < pos_tol)
            acc_dict[label_name + '_acc'] = np.sum(correct) / total_counted

        if label_name in ROT_LABEL_NAMES:
            # convert rad
            targ_rot = labels[label_name] * np.pi/180
            pred_rot = predictions[label_name] * np.pi/180

            # make sure same range
            targ_rot[targ_rot < 0] += 2 * np.pi
            pred_rot[pred_rot < 0] += 2 * np.pi

            # calc error (TODO: fix to catch cases on boundary of [0, 360])
            abs_err = torch.abs(targ_rot - pred_rot).detach().cpu().numpy()
            correct = (abs_err < (rot_tol * np.pi/180))
            acc_dict[label_name + '_acc'] = np.sum(correct) / total_counted

        overall_correct = overall_correct & correct

    # count where all predictions are correct for overall accuracy
    overall_correct = np.sum(overall_correct)
    acc_dict['overall_acc'] = overall_correct / total_counted

    return acc_dict
