"""
Evaluate a trained network on a testset
"""
import torch

import visualize


def predict_and_display_sample(net, classes, test_loader, device='cpu'):
    dataiter = iter(test_loader)
    images, true_labels = next(dataiter)

    net.eval()
    with torch.no_grad():
        outputs = net(images.to(device))
    net.train()
    _, predicted_labels = torch.max(outputs.data.cpu(), 1)

    print(f"{'Ground Truth:':<15}", end='')
    visualize.print_labels(true_labels, classes)
    print(f"{'Predicted:':<15}", end='')
    visualize.print_labels(predicted_labels, classes)

    visualize.show_images(images.detach().cpu())


def calculate_and_display_test_accuracy(net, test_loader, device='cpu'):
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    net.train()
    accuracy = 100.0 * (float(correct) / total)

    print(f'Accuracy on the testset: {accuracy:.2f} %')
