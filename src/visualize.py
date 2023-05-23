"""
Visualization functionality
"""

import torchvision

import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    image = image / 2 + 0.5
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_images(images):
    show_image(torchvision.utils.make_grid(images))


def print_labels(labels, classes):
    print(' '.join(f'{classes[label]:5s}' for label in labels))


def plot_learning_curve(train_losses, val_losses):
    fig, ax = plt.subplots()

    x = np.arange(1, len(train_losses))
    ax.set_title("learning curve")
    ax.set_xlabel("epoch")
    ax.locator_params(axis='x', integer=True)
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.plot(x, train_losses[1:], c='blue', label='train')
    ax.plot(x, val_losses[1:], c='blue', label='val', linestyle='--')
    ax.legend()
    plt.show()
    return fig
