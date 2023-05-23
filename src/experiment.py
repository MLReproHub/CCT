"""
The actual experiments that will be run using different models. This
encompasses training and testing.
"""
import os
import pathlib
import random
import sys
from typing import Tuple
from datetime import datetime

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import random_split, DataLoader

import dataset
import evaluate
import model
import train
from visualize import plot_learning_curve

# set base path
BASE_PATH = pathlib.Path(__file__).parent.parent.resolve()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)


def load_or_init_model(filename, model_name, model_params, device='cpu'):
    # Instantiate model
    model_class = getattr(model, model_name)
    net = model_class(**model_params)
    # Load checkpoint or initialize model weights
    checkpoints_dir = os.path.join(BASE_PATH, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    filename = os.path.join(checkpoints_dir, os.path.basename(filename))
    if os.path.isfile(filename) and os.path.exists(filename):
        net.load_state_dict(torch.load(filename, map_location='cpu'))
        net.trained = True
    else:
        net.apply(init_weights)
    return net.to(device), filename


def load_config(config_filename: str) -> dict:
    # Find path to configuration files
    config_filename_full_path = os.path.join(BASE_PATH, 'configs', config_filename + '.yaml')
    if not os.path.exists(config_filename_full_path) or not os.path.isfile(config_filename_full_path):
        print(f"Cannot find configuration file:\n\t{config_filename_full_path}", file=sys.stderr)
        exit(1)
    # Load config from file
    with open(config_filename_full_path) as yaml_fp:
        config = yaml.load(yaml_fp, yaml.FullLoader)
    return config


def load_data(dl_conf: dict, device, num_workers) -> \
        Tuple[dataset.Cifar10Dataloader, dataset.Cifar10Dataloader, dataset.Cifar10Dataloader, Tuple[str]]:
    ds_class = getattr(dataset, f'{dl_conf["which"]}Dataset')
    dl_class = getattr(dataset, f'{dl_conf["which"]}Dataloader')
    ds_params = dl_conf['params'] if 'params' in dl_conf.keys() else {}
    ds_train = ds_class(train=True, **ds_params)
    ds_val = ds_class(train=True)
    ds_train_train, _ = random_split(ds_train, [45000, 5000])
    _, ds_train_val = random_split(ds_val, [45000, 5000])

    dl_train = DataLoader(ds_train_train, batch_size=dl_conf['batch_size'], shuffle=True, pin_memory=device == 'cuda',
                          drop_last=True, num_workers=num_workers)
    dl_val = DataLoader(ds_train_val, batch_size=dl_conf['batch_size'], shuffle=False, pin_memory=device == 'cuda',
                        drop_last=False, num_workers=num_workers)
    dl_test = dl_class(train=False, batch_size=dl_conf['batch_size'], shuffle=False, pin_memory=device == 'cuda',
                       drop_last=False, num_workers=num_workers)

    classes = ds_class.CLASS_NAMES

    return dl_train, dl_val, dl_test, classes


def mkl_conflict_workaround():
    # Getting some error with intel MKL due to conflicting installations in the conda env
    # Dirty workaround from https://github.com/explosion/spaCy/issues/7664#issuecomment-825501808
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def ssl_cert_workaround():
    import ssl
    # Workaround to a ssl certificate error
    # https://stackoverflow.com/a/71491515
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa


def main(argv):
    # Parse command line arguments
    num_args = len(argv)
    if num_args < 2 or num_args > 4:
        print("Wrong number of command line arguments", file=sys.stderr)
        print(f"\nUsage: {argv[0]} CONFIG_FILENAME [NUM_WORKERS] [-noplots]")
        print("\nwhere")
        print(f"\tCONFIG_FILENAME\tis a .yaml file relative to\n\t\t\t{os.path.join(BASE_PATH, 'config')}\n")
        print(f"\t\t\tor two comma-separated .yaml files relative to\n\t\t\t{os.path.join(BASE_PATH, 'config')}\n")
        print(f"\tNUM_WORKERS\tis the number of workers for the data loaders.")
        print("\t\t\tIf not specified, 1 is used as default.\n")
        print(f"\t-noplots\tdisables all plotting of results.")
        exit(1)

    config_filename = argv[1]

    num_workers = 1
    no_plots = False

    def parse_arg(arg):
        nonlocal num_workers, no_plots
        if arg == '-noplots':
            no_plots = True
        else:
            num_workers = int(arg)

    for arg in argv[2:]:
        parse_arg(arg)

    # Load configuration
    config_filenames = config_filename.split(',', maxsplit=1)
    configs = [load_config(config_filename) for config_filename in config_filenames]
    if len(configs) == 1:
        config = configs[0]
        dl_config = config.pop('dl')
        config_filename = config_filenames[0]
    else:
        dl_config = configs[0]
        config = configs[1]
        config_filename = f"{config_filenames[1].replace('model/', '')}__{config_filenames[0].replace('dl/cifar10_', '')}"

    # Choosing pytorch device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize random number seed
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    # Instantiate dataloaders
    dl_train, dl_val, dl_test, classes = load_data(dl_config, device, num_workers)

    # Instantiate model
    model_name = config['model']['which']
    model_params = config['model']['params']
    net, chkpt_filepath = load_or_init_model(config_filename + '.pth', model_name, model_params, device=device)

    # Train model
    if not net.trained:
        oc = config['optim']
        num_epochs = config['num_epochs']
        train_losses, val_losses = train.train(net, dl_train, dl_val, epochs=num_epochs, device=device,
                                               optim_type=oc['which'], **oc['params'])

        # Save model
        torch.save(net.state_dict(), chkpt_filepath)

        # Save training stats
        stats_path = os.path.join(BASE_PATH, 'stats')
        os.makedirs(stats_path, exist_ok=True)
        stats_filename = f"{stats_path}/{config_filename}_{datetime.now().strftime('%H%M%S-%d%m')}.yaml"
        with open(stats_filename, 'w') as file:
            yaml.dump({'train_losses': train_losses, 'val_losses': val_losses}, file)

        # Plot training stats and save
        if not no_plots:
            fig = plot_learning_curve(train_losses, val_losses)
            graphics_path = os.path.join(BASE_PATH, 'graphics')
            os.makedirs(graphics_path, exist_ok=True)
            fig.savefig(f"{graphics_path}/{config_filename}_{datetime.now().strftime('%H%M%S-%d%m')}")

    # Evaluate model
    if not no_plots:
        evaluate.predict_and_display_sample(net, classes, dl_test, device=device)
    evaluate.calculate_and_display_test_accuracy(net, dl_test, device=device)


if __name__ == '__main__':
    main(sys.argv)
