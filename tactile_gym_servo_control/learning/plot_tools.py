import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES
from tactile_gym.utils.general_utils import load_json_obj


sns.set_theme(style="darkgrid")
model_path = os.path.join(os.path.dirname(__file__), '../learned_models')


def plot_error(
    pred_df,
    targ_df,
    err_df,
    targ_label_names,
    pose_limits=None,
    fig=None,
    axs=None,
    save_file=None,
    show_plot=True,
):

    if fig is None and axs is None:
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 10))
        fig.subplots_adjust(wspace=0.3)

    n_smooth = int(pred_df.shape[0] / 20)

    for i, ax in enumerate(axs.flat):

        pose_label = POSE_LABEL_NAMES[i]

        if pose_label not in targ_label_names:
            continue

        # sort all dfs by target
        targ_df = targ_df.sort_values(by=pose_label)

        pred_df = pred_df.assign(temp=targ_df[pose_label])
        pred_df = pred_df.sort_values(by='temp')
        pred_df = pred_df.drop('temp', axis=1)

        err_df = err_df.assign(temp=targ_df[pose_label])
        err_df = err_df.sort_values(by='temp')
        err_df = err_df.drop('temp', axis=1)

        ax.scatter(
            targ_df[pose_label],
            pred_df[pose_label],
            s=1,
            c=err_df[pose_label], cmap="inferno"
        )

        ax.plot(
            targ_df[pose_label].rolling(n_smooth).mean(),
            pred_df[pose_label].rolling(n_smooth).mean(),
            linewidth=2,
            c="red"
        )

        ax.set(xlabel=f"target {pose_label}", ylabel=f"predicted {pose_label}")

        if pose_limits is not None:
            ax.set_xlim(pose_limits[0][i], pose_limits[1][i])
            ax.set_ylim(pose_limits[0][i], pose_limits[1][i])

        ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[pose_label].mean()), transform=ax.transAxes)
        ax.grid(True)

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_training(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    n_epochs,
    fig=None,
    axs=None,
    save_file=None,
    show_plot=True,
):

    # convert lists to arrays
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)

    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    x_data = np.arange(n_epochs)
    train_window_size = int(np.floor(train_loss.shape[0] / n_epochs))
    val_window_size = int(np.floor(val_loss.shape[0] / n_epochs))

    train_loss = train_loss.reshape(-1, train_window_size)
    val_loss = val_loss.reshape(-1, val_window_size)
    train_acc = train_acc.reshape(-1, train_window_size)
    val_acc = val_acc.reshape(-1, val_window_size)

    lo_bound = np.clip(train_loss.mean(axis=1) - train_loss.std(axis=1), train_loss.min(axis=1), train_loss.max(axis=1))
    up_bound = np.clip(train_loss.mean(axis=1) + train_loss.std(axis=1), train_loss.min(axis=1), train_loss.max(axis=1))
    axs[0].plot(x_data, train_loss.mean(axis=1), color='r', alpha=1.0)
    axs[0].fill_between(
        x_data,
        lo_bound,
        up_bound,
        color='r',
        alpha=0.25
    )

    lo_bound = np.clip(val_loss.mean(axis=1) - val_loss.std(axis=1), val_loss.min(axis=1), val_loss.max(axis=1))
    up_bound = np.clip(val_loss.mean(axis=1) + val_loss.std(axis=1), val_loss.min(axis=1), val_loss.max(axis=1))
    axs[0].plot(x_data, val_loss.mean(axis=1), color='b', alpha=1.0)
    axs[0].fill_between(
        x_data,
        lo_bound,
        up_bound,
        color='b',
        alpha=0.25
    )

    axs[0].set_yscale('log')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    lo_bound = np.clip(train_acc.mean(axis=1) - train_acc.std(axis=1), train_acc.min(axis=1), train_acc.max(axis=1))
    up_bound = np.clip(train_acc.mean(axis=1) + train_acc.std(axis=1), train_acc.min(axis=1), train_acc.max(axis=1))
    axs[1].plot(x_data, train_acc.mean(axis=1), color='r', alpha=1.0)
    axs[1].fill_between(
        x_data,
        lo_bound,
        up_bound,
        color='r',
        alpha=0.25,
        label='_nolegend_'
    )

    lo_bound = np.clip(val_acc.mean(axis=1) - val_acc.std(axis=1), val_acc.min(axis=1), val_acc.max(axis=1))
    up_bound = np.clip(val_acc.mean(axis=1) + val_acc.std(axis=1), val_acc.min(axis=1), val_acc.max(axis=1))
    axs[1].plot(x_data, val_acc.mean(axis=1), color='b', alpha=1.0)
    axs[1].fill_between(
        x_data,
        lo_bound,
        up_bound,
        color='b',
        alpha=0.25,
        label='_nolegend_'
    )

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    plt.legend(['Train', 'Val'])

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    if show_plot:
        plt.show()


if __name__ == '__main__':

    task = 'surface_3d'
    # task = 'edge_2d'
    # task = 'edge_3d'
    # task = 'edge_5d'

    # model for loading
    save_dir_name = os.path.join(
        model_path,
        task,
        'tap',
    )

    train_loss = np.load(os.path.join(save_dir_name, 'train_loss.npy'))
    val_loss = np.load(os.path.join(save_dir_name, 'val_loss.npy'))
    train_acc = np.load(os.path.join(save_dir_name, 'train_acc.npy'))
    val_acc = np.load(os.path.join(save_dir_name, 'val_acc.npy'))

    learning_params = load_json_obj(os.path.join(save_dir_name, 'learning_params'))

    plot_training(
        train_loss, val_loss,
        train_acc, val_acc,
        n_epochs=learning_params['epochs'],
        save_file=os.path.join(save_dir_name, 'training_curves.png')
    )
