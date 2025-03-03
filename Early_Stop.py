class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pth'):
        """
        Early stopping based on validation loss.

        Parameters:
        - patience: how many epochs to wait after the last improvement before stopping.
        - min_delta: minimum change to count as an improvement.
        - path: where to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0  # Counts the number of epochs since the last improvement
        self.best_score = None  # Best validation loss
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss):
        """
        This function is called every epoch to check whether early stopping should occur.
        """
        if self.best_score is None:
            self.best_score = val_loss
            # self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            # self.save_checkpoint(model)
            self.counter = 0  # Reset counter after an improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    # def save_checkpoint(self, model):
    #     """
    #     Save the model's state_dict if the current model is the best one.
    #     """
    #     self.best_model_wts = model.state_dict()