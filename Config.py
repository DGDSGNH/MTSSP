import torch
import numpy as np
RANDOM_SEED = 12345
batch_size = 256
input_size = 63
embed_dim = 32
num_layers = 1
hidden_size = 32
workers = 2
learning_rate = 1e-2
epochs = 50
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
datatype = "mimic3"

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=4, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.state = {}
        self.save_model_dir = ""

    def __call__(self, val_loss, model, state, save_model_dir):

        score = val_loss
        self.state = state
        self.save_model_dir = save_model_dir
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            print(f'Best Score:{self.best_score}')
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        # torch.save(state, save_model_dir)
        torch.save(self.state, self.save_model_dir)                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss