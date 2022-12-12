import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch_model_summary import summary
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from tactile_gym.utils.general_utils import save_json_obj, check_dir

from tactile_gym_servo_control.learning.learning_utils import get_pose_limits
from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.learning.learning_utils import encode_pose
from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.learning_utils import acc_metric
from tactile_gym_servo_control.learning.learning_utils import err_metric
from tactile_gym_servo_control.learning.plot_tools import plot_training
from tactile_gym_servo_control.learning.plot_tools import plot_error
from tactile_gym_servo_control.learning.networks import CNN, weights_init_normal
from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator
from tactile_gym_servo_control.learning.test_cnn import test_cnn


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = os.path.join(os.path.dirname(__file__), '../data/')
model_path = os.path.join(os.path.dirname(__file__), '../learned_models')

# tolerances for accuracy metric
POS_TOL = 0.25  # mm
ROT_TOL = 1.0  # deg


def train_cnn(task, learning_params, image_processing_params, augmentation_params, device='cpu'):

    # set save dir
    save_dir_name = os.path.join(
        model_path,
        task,
        'tap',
    )

    # check save dir exists
    check_dir(save_dir_name)
    os.makedirs(save_dir_name, exist_ok=True)

    # save parameters
    save_json_obj(learning_params, os.path.join(save_dir_name, 'learning_params'))
    save_json_obj(image_processing_params, os.path.join(save_dir_name, 'image_processing_params'))
    save_json_obj(augmentation_params, os.path.join(save_dir_name, 'augmentation_params'))

    # set the correct accuracy metric and label generator
    out_dim, label_names = import_task(task)

    # data dir
    # can specifiy multiple directories that get combined in generator
    train_data_dirs = [
        os.path.join(data_path, task, 'tap', 'train')
    ]
    train_pose_limits = get_pose_limits(train_data_dirs, save_dir_name)

    validation_data_dirs = [
        os.path.join(data_path, task, 'tap', 'val')
    ]

    # set generators and loaders
    generator_args = {**image_processing_params, **augmentation_params}
    train_generator = ImageDataGenerator(data_dirs=train_data_dirs, **generator_args)
    val_generator = ImageDataGenerator(data_dirs=validation_data_dirs, **image_processing_params)

    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # create model
    model = CNN(out_dim, image_processing_params['dims'], learning_params).to(device)
    model.apply(weights_init_normal)
    print(summary(
        model,
        torch.zeros((1, 1, *image_processing_params['dims'])).to(device),
        show_input=True
    ))

    # define optimizer and loss
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_params['lr'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )

    def run_epoch(loader, n_batches, training=True):

        epoch_batch_loss = []
        epoch_batch_acc = []
        acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
        err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

        # complete dateframe of predictions and targets
        if not training:
            pred_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
            targ_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

        for batch in loader:

            # get inputs
            inputs, labels_dict = batch['images'], batch['labels']

            # wrap them in a Variable object
            inputs = Variable(inputs).float().to(device)

            # get labels
            labels = encode_pose(labels_dict, label_names, train_pose_limits, device)

            # set the parameter gradients to zero
            if training:
                optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)

            if training:
                loss_size.backward()
                optimizer.step()

            # count correct for accuracy metric
            predictions_dict = decode_pose(outputs, label_names, train_pose_limits)

            if not training:
                # append predictions and labels to dataframes
                batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
                batch_targ_df = pd.DataFrame.from_dict(labels_dict)
                pred_df = pd.concat([pred_df, batch_pred_df])
                targ_df = pd.concat([targ_df, batch_targ_df])

            # get errors and accuracy
            batch_err_df = err_metric(labels_dict, predictions_dict, label_names)
            batch_acc_df = acc_metric(labels_dict, predictions_dict, label_names, POS_TOL, ROT_TOL)

            # append error to dataframe
            err_df = pd.concat([err_df, batch_err_df])
            acc_df = pd.concat([acc_df, batch_acc_df])

            # statistics
            epoch_batch_loss.append(loss_size.item())
            epoch_batch_acc.append(acc_df['overall_acc'].mean())

        # reset indices to be 0 -> test set size
        acc_df = acc_df.reset_index(drop=True).fillna(0.0)
        err_df = err_df.reset_index(drop=True).fillna(0.0)

        if not training:
            pred_df = pred_df.reset_index(drop=True).fillna(0.0)
            targ_df = targ_df.reset_index(drop=True).fillna(0.0)
            return epoch_batch_loss, epoch_batch_acc, acc_df, err_df, pred_df, targ_df
        else:
            return epoch_batch_loss, epoch_batch_acc, acc_df, err_df

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # for saving best model
    highest_val_acc = 0

    if learning_params['plot_during_training']:
        # create figures for updating
        plt.ion()
        train_fig, train_axs = plt.subplots(1, 2, figsize=(12, 4))
        err_fig, err_axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
        plt.show()
        plt.pause(0.01)

    def update_train_plot(train_loss, val_loss, train_acc, val_acc, n_epochs):

        # clear axis
        for ax in train_axs.flat:
            ax.clear()

        # plot training
        plot_training(
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            n_epochs=n_epochs,
            fig=train_fig,
            axs=train_axs,
            save_file=os.path.join(save_dir_name, 'training_curves.png'),
            show_plot=False
        )

        for ax in train_axs.flat:
            ax.relim()
            ax.set_xlim([0, learning_params['epochs']])
            ax.autoscale_view(True, True, True)

        # show results
        train_fig.canvas.draw()
        plt.pause(0.01)

    def update_error_plot(
        pred_df,
        targ_df,
        err_df,
    ):

        # clear axis
        for ax in err_axs.flat:
            ax.clear()

        # plot error
        plot_error(
            pred_df,
            targ_df,
            err_df,
            label_names,
            train_pose_limits,
            fig=err_fig,
            axs=err_axs,
            save_file=os.path.join(save_dir_name, 'error_plot.png'),
            show_plot=False
        )

        # show results
        err_fig.canvas.draw()
        plt.pause(0.01)

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            train_epoch_loss, train_epoch_acc, train_acc_df, train_err_df = run_epoch(
                train_loader, n_train_batches, training=True)

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc, val_acc_df, val_err_df, val_pred_df, val_targ_df = run_epoch(
                val_loader, n_val_batches, training=False)
            model.train()

            # append loss and acc
            train_loss.extend(train_epoch_loss)
            train_acc.extend(train_epoch_acc)
            val_loss.extend(val_epoch_loss)
            val_acc.extend(val_epoch_acc)

            # print metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch+1))
            print("")

            print("Train Metrics")
            print("train_acc: {:.6f}".format(np.mean(train_epoch_acc)))
            print(train_acc_df[label_names].mean())
            print("train_err: {:.6f}".format(np.mean(train_epoch_loss)))
            print(train_err_df[label_names].mean())

            print("")
            print("Validation Metrics")
            print("val_acc: {:.6f}".format(np.mean(val_epoch_acc)))
            print(val_acc_df[label_names].mean())
            print("val_err: {:.6f}".format(np.mean(val_epoch_loss)))
            print(val_err_df[label_names].mean())
            print("")

            # update plots
            if learning_params['plot_during_training']:
                update_train_plot(train_loss, val_loss, train_acc, val_acc, epoch)
                update_error_plot(val_pred_df, val_targ_df, val_err_df)

            # save the model with lowest val loss
            if np.mean(val_epoch_acc) > highest_val_acc:
                highest_val_acc = np.mean(val_epoch_acc)
                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir_name, 'best_model.pth')
                )

            # decay the lr
            lr_scheduler.step(np.mean(val_epoch_loss))

            # update epoch progress bar
            pbar.update(1)

    print("Training finished, took {:.6f}s".format(time.time() - training_start_time))

    # save the final
    torch.save(
        model.state_dict(),
        os.path.join(save_dir_name, 'final_model.pth')
    )

    # convert lists to arrays
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)

    # save loss and acc lists
    np.save(os.path.join(save_dir_name, 'train_loss.npy'), train_loss)
    np.save(os.path.join(save_dir_name, 'val_loss.npy'), val_loss)
    np.save(os.path.join(save_dir_name, 'train_acc.npy'), train_acc)
    np.save(os.path.join(save_dir_name, 'val_acc.npy'), val_acc)

    # plot progress
    plot_training(
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        n_epochs=learning_params['epochs'],
        save_file=os.path.join(save_dir_name, 'training_curves.png'),
        show_plot=True
    )

    test_cnn(
        task,
        device
    )


if __name__ == "__main__":

    # tasks = ['surface_3d']
    # tasks = ['edge_2d']
    # tasks = ['edge_3d']
    # tasks = ['edge_5d']
    tasks = ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']

    # Parameters
    learning_params = {
        'batch_size': 128,
        'epochs': 250,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'dropout': 0.0,
        'shuffle': True,
        'n_cpu': 8,
        'apply_batchnorm': True,
        'plot_during_training': False,
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
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    device = 'cuda'
    # device = 'cpu'

    for task in tasks:
        train_cnn(task, learning_params, image_processing_params, augmentation_params, device)
