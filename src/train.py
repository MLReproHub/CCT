"""
Training a network
Displays statistics during the training
"""
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import SGD
from tqdm.autonotebook import tqdm

from pytorch_transformers.optimization import WarmupCosineSchedule

import dataset


def train(net, dl_train, dl_val, epochs: int = 2, device='cpu', optim_type: str = 'AdamW', **optim_kwargs):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = getattr(optim, optim_type)(net.parameters(), **optim_kwargs)
    lr = optim_kwargs["lr"]
    # https://huggingface.co/transformers/v1.0.0/model_doc/overview.html#learning-rate-schedules
    batches_per_epoch = dl_train.sampler.num_samples // dl_train.batch_size
    n_updates = epochs * batches_per_epoch
    # TODO epoch-relative and absolute warmup steps to config
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10 * batches_per_epoch, t_total=n_updates)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10 * batches_per_epoch, t_total=n_updates)
    train_losses = [0.0, ]
    val_losses = [0.0, ]
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_running_loss = 0.0
        pbar = tqdm(dl_train)
        i = 0
        for i, data in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize + update lr
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            train_running_loss += loss.item()
            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f'e:{epoch + 1:3d}/{epochs:3d}|tl:{train_losses[-1]:.3f}|vl:{val_losses[-1]:.3f}|'
                                 f'lr:{lr:.3E}')
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
        train_losses.append(train_running_loss / (i + 1))

        # Update validation loss
        val_running_loss = 0.0
        with torch.no_grad():
            net.eval()
            val_i = 0
            for val_i, val_data in enumerate(dl_val):
                val_inp, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_out = net(val_inp)
                val_loss = criterion(val_out, val_labels)
                val_running_loss += val_loss.item()
            val_losses.append(val_running_loss / (val_i + 1))
            net.train()
    print('Finished Training')

    return train_losses, val_losses


def plot_lr(dl_train, epochs: int, lr: float):
    net = nn.Conv2d(3, 1, kernel_size=3)
    optimizer = SGD(net.parameters(), lr=lr)
    batches_per_epoch = dl_train.sampler.num_samples // dl_train.batch_size
    n_updates = epochs * batches_per_epoch
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10 * batches_per_epoch, t_total=n_updates)
    lrs = []
    lrs_e = []
    for e in range(epochs):
        for b in range(batches_per_epoch):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            lrs_e.append(e + b / batches_per_epoch)
    plt.plot(lrs_e, lrs)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title(f'Learning Rate Scheduling for {epochs} epochs')
    plt.tight_layout()
    plt.savefig('lrs.pdf')
    plt.show()


if __name__ == '__main__':
    dl_train = dataset.Cifar10Dataloader(train=True, batch_size=128, shuffle=True)
    plot_lr(dl_train, epochs=200, lr=1e-4)
