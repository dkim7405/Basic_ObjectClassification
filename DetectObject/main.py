from utils.training_config import SystemConfiguration, TrainingConfiguration
from data.data_loader import get_data
from models.my_model import MyModel
from train import main
import torch

if __name__ == '__main__':
    # System configuration
    system_configuration = SystemConfiguration()

    # Training configuration
    training_configuration = TrainingConfiguration(
        batch_size=128,
        epochs_count=20,
        learning_rate=0.02,
        log_interval=100,
        test_interval=1,
        data_root="images",
        num_workers=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run the main training process
    model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = main(system_configuration, training_configuration)
