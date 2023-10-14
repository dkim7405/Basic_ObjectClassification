import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.training_config import TrainingConfiguration, SystemConfiguration
from models.my_model import MyModel
from data.data_loader import get_data
from utils.training_config import save_model

def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

def train(train_config: TrainingConfiguration, model: nn.Module, optimizer: optim.Optimizer,
          train_loader: torch.utils.data.DataLoader, epoch_idx: int) -> None:
    # Training logic goes here
    pass

def validate(train_config: TrainingConfiguration, model: nn.Module,
             test_loader: torch.utils.data.DataLoader) -> float:
    # Validation logic goes here
    pass

def main(system_config: SystemConfiguration, training_config: TrainingConfiguration):
    setup_system(system_config)
    batch_size_to_set = training_config.batch_size
    num_workers_to_set = training_config.num_workers
    epoch_num_to_set = training_config.epochs_count

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        num_workers_to_set = 2

    train_loader, test_loader = get_data(
        batch_size=training_config.batch_size,
        data_root=training_config.data_root,
        num_workers=num_workers_to_set
    )

    training_config = TrainingConfiguration(
        device=device,
        num_workers=num_workers_to_set
    )

    model = MyModel()
    model.to(training_config.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=training_config.learning_rate
    )

    best_loss = torch.tensor(np.inf)
    best_accuracy = torch.tensor(0)

    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])

    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])

    t_begin = time.time()
    for epoch in range(training_config.epochs_count):
        train_loss, train_acc = train(training_config, model, optimizer, train_loader, epoch)

        epoch_train_loss = np.append(epoch_train_loss, [train_loss])

        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_config.epochs_count - elapsed_time

        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )

        if epoch % training_config.test_interval == 0:
            current_loss, current_accuracy = validate(training_config, model, test_loader)

            epoch_test_loss = np.append(epoch_test_loss, [current_loss])

            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])

            if current_loss < best_loss:
                best_loss = current_loss

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                print('Accuracy improved, saving the model.\n')
                save_model(model, training_config.device)
    
    print("Total time: {:.2f}, Best Loss: {:.3f}, Best Accuracy: {:.3f}".format(time.time() - t_begin, best_loss,
                                                                                best_accuracy))

    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc

if __name__ == "__main__":
    system_config = SystemConfiguration()
    training_config = TrainingConfiguration()
    model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = main(system_config, training_config)
