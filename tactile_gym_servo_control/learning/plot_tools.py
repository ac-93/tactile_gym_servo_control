import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training(train_loss, validation_loss, train_acc, validation_acc, save_file=None):

    x_data = range(len(train_loss))

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].plot(x_data, train_loss, color='r', alpha=1.0)
    axs[0].plot(x_data, validation_loss, color='b', alpha=1.0)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(x_data, train_acc, color='r', alpha=1.0)
    axs[1].plot(x_data, validation_acc, color='b', alpha=1.0)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    plt.legend(['Train', 'Val'])

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    plt.show()


def plot_error(label_array, pred_array, label_names, save_file=None):
    abs_err = np.abs(label_array-pred_array)
    mae_dict = {k + '_mae': np.mean(abs_err[:, i]) for (i, k) in enumerate(label_names)}
    std_dict = {k + '_std': np.std(abs_err[:, i]) for (i, k) in enumerate(label_names)}

    fig, ax = plt.subplots()
    ax.bar(mae_dict.keys(), mae_dict.values(), yerr=std_dict.values(), capsize=5)
    ax.set_xticklabels(mae_dict.keys(), rotation=45)
    ax.set_ylabel('Mean Absolute Error')
    plt.show()

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    mae_dict['overall_mae'] = np.mean(abs_err)
    std_dict['overall_std'] = np.mean(np.std(abs_err, axis=0))

    return mae_dict, std_dict


if __name__ == '__main__':

    task = 'surface_3d'
    data_collection = 'tap'

    # set image size
    dims = [128, 128]

    # model for loading
    save_dir_name = os.path.join(
        'saved_models',
        task,
        data_collection,
    )

    loss_arr = np.load(os.path.join(
        save_dir_name,
        'loss_arr.npy'
    ))
    acc_arr = np.load(os.path.join(
        save_dir_name,
        'acc_arr.npy'
    ))

    plot_training(loss_arr[0, :], loss_arr[1, :], acc_arr[0, :], acc_arr[1, :],
                  save_file=os.path.join(save_dir_name, 'training_curves.png'))
