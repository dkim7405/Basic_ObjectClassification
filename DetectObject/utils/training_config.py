from dataclasses import dataclass

@dataclass
class SystemConfiguration:
    '''
    Describes the common system settings needed for reproducible training
    '''
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make CuDNN deterministic (reproducible training)

@dataclass
class TrainingConfiguration:
    '''
    Describes the configuration of the training process
    '''
    batch_size: int = 128  # amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 20  # number of times the whole dataset will be passed through the network
    learning_rate: float = 0.02  # determines the speed of network's weights update
        
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get validation loss at each epoch
    data_root: str = "images"  # folder to save data
    num_workers: int = 16  # number of concurrent processes to use to prepare data
    device: str = 'cuda'  # device to use for training
