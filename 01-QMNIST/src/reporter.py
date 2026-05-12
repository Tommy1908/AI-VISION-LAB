class ExperimentTracker:
    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.config = config
        self.history = {"train_loss": [], "test_loss": [], "test_acc": [], "gen_gap": []}


    def log_epoch(self, train_loss, test_loss, test_acc):
        gap = test_loss - train_loss
        
        self.history["train_loss"].append(train_loss)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["gen_gap"].append(gap)


    def save_local_report(self):
        return

    def finish(self):
        self.save_local_report()
        return self.history

def plot_project_master_report(project_name, all_results):
    return