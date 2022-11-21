import os
import time
import numpy as np
from pprint import pprint
from tqdm import tqdm
from pytorch_model_summary import summary
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch

from tactile_gym.utils.general_utils import save_json_obj, check_dir

from tactile_gym_servo_control.learning.learning_utils import get_pose_limits
from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.learning.learning_utils import encode_pose
from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import acc_metric
from tactile_gym_servo_control.learning.plot_tools import plot_training
from tactile_gym_servo_control.learning.networks import CNN, weights_init_normal
from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_cnn(device='cpu'):

    # task = 'surface_3d'
    # task = 'edge_2d'
    # task = 'edge_3d'
    task = 'edge_5d'

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

    data_collection = 'tap'

    # set save dir
    save_dir_name = os.path.join(
        os.path.dirname(__file__),
        'saved_models',
        task,
        data_collection,
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
    training_data_dirs = [
        os.path.join(os.path.dirname(__file__), '../data/', task, 'tap', 'train')
    ]
    train_pose_limits = get_pose_limits(training_data_dirs, save_dir_name)

    validation_data_dirs = [
        os.path.join(os.path.dirname(__file__), '../data/', task, 'tap', 'val')
    ]

    # set generators and loaders
    generator_args = {**image_processing_params, **augmentation_params}
    training_generator = ImageDataGenerator(data_dirs=training_data_dirs, **generator_args)
    val_generator = ImageDataGenerator(data_dirs=validation_data_dirs, **image_processing_params)

    training_loader = torch.utils.data.DataLoader(
        training_generator,
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

    n_train_batches = len(training_loader)
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

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss_list = []
    train_acc_list = []
    validation_loss_list = []
    validation_acc_list = []

    # for saving best model
    highest_val_acc = 0

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(learning_params['epochs']):

            # ========= Train =========
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            train_acc_dict = {k + '_acc': 0.0 for k in label_names}
            for i, train_batch in enumerate(training_loader, 0):

                # get inputs
                inputs, labels_dict = train_batch['images'], train_batch['labels']

                # wrap them in a Variable object
                inputs = Variable(inputs).float().to(device)

                # get labels
                labels = encode_pose(labels_dict, label_names, train_pose_limits, device)

                # set the parameter gradients to zero
                optimizer.zero_grad()

                # forward pass, backward pass, optimize
                outputs = model(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                # count correct for accuracy metric
                predictions_dict = decode_pose(outputs, label_names, train_pose_limits)
                acc_dict = acc_metric(labels_dict, predictions_dict, label_names)

                # print statistics
                epoch_train_acc += acc_dict['overall_acc']
                epoch_train_loss += loss_size.item()

                # sum acc dict for reporting
                for k in train_acc_dict:
                    train_acc_dict[k] += acc_dict[k]

            # calc epoch loss and acc
            epoch_train_loss /= n_train_batches
            epoch_train_acc /= n_train_batches
            train_acc_dict = {k: v / n_train_batches for k, v in train_acc_dict.items()}

            # print train metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch+1))
            print("train_loss: {:.4f}".format(epoch_train_loss))
            print("train_acc: {:.4f}".format(epoch_train_acc))
            pprint(train_acc_dict, sort_dicts=False)

            # append training loss and acc
            train_loss_list.append(epoch_train_loss)
            train_acc_list.append(epoch_train_acc)

            # ========= Validation =========
            model.eval()  # turn off batchnorm/dropout

            total_val_loss = 0.0
            total_val_acc = 0.0
            val_acc_dict = {k + '_acc': 0.0 for k in label_names}
            for val_batch in val_loader:

                # get inputs
                inputs, labels_dict = val_batch['images'], val_batch['labels']

                # wrap them in a Variable object
                inputs = Variable(inputs).float().to(device)

                # get labels
                labels = encode_pose(labels_dict, label_names, train_pose_limits, device)

                # forward pass
                val_outputs = model(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

                # count correct for accuracy metric
                predictions_dict = decode_pose(val_outputs, label_names, train_pose_limits)
                acc_dict = acc_metric(labels_dict, predictions_dict, label_names)
                total_val_acc += acc_dict['overall_acc']

                # sum acc dict for reporting
                for k in val_acc_dict:
                    val_acc_dict[k] += acc_dict[k]

            # calc loss and acc
            val_loss = total_val_loss / n_val_batches
            val_acc = total_val_acc / n_val_batches
            val_acc_dict = {k: v / n_val_batches for k, v in val_acc_dict.items()}

            # disp metrics
            validation_loss_list.append(val_loss)
            validation_acc_list.append(val_acc)

            # print val metrics
            print("val_loss: {:.4f}".format(val_loss))
            print("val_acc: {:.4f}".format(val_acc))
            pprint(val_acc_dict, sort_dicts=False)

            # turn back to training model
            model.train()

            # save the model with lowest val loss
            if val_acc > highest_val_acc:
                highest_val_acc = val_acc
                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir_name, 'best_model.pth')
                )

            # decay the lr
            lr_scheduler.step(val_loss)
            print('')

            # update epoch progress bar
            pbar.update(1)

    print("Training finished, took {:.4f}s".format(time.time() - training_start_time))

    # save the final
    torch.save(
        model.state_dict(),
        os.path.join(save_dir_name, 'final_model.pth')
    )

    # save loss and acc lists
    loss_arr = np.stack([train_loss_list, validation_loss_list])
    acc_arr = np.stack([train_acc_list, validation_acc_list])
    np.save(os.path.join(save_dir_name, 'loss_arr.npy'), loss_arr)
    np.save(os.path.join(save_dir_name, 'acc_arr.npy'), acc_arr)

    # plot progress
    plot_training(
        train_loss_list,
        validation_loss_list,
        train_acc_list,
        validation_acc_list,
        save_file=os.path.join(save_dir_name, 'training_curves.png')
    )

    return 0


if __name__ == "__main__":
    device = 'cuda'
    # device = 'cpu'
    train_cnn(device)
