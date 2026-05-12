import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, config, tracker=None):
        self.model = model
        self.config = config
        self.tracker = tracker
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=config["lr"]
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = config["device"]

    def fit(self, train_loader, test_loader):
        #tracker = ExperimentTracker(self.config)

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss, _ = run_epoch(
                train_loader, self.model, self.loss_fn, self.optimizer, self.device
            )
            #Run with test to see each epoch performance, but without training
            test_loss, test_acc = run_epoch(
                test_loader, self.model, self.loss_fn, device=self.device
            )
            #tracker.log_epoch(epoch, train_loss, test_loss, test_acc)
            if self.tracker:
                self.tracker.log_epoch(train_loss, test_loss, test_acc)

            #if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.4f} "
                    f"| Test Loss: {test_loss:.4f} | Acc: {100*test_acc:.1f}%"
                )

        return


def run_epoch(loader, model, loss_fn, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0 # Sums the average loss of each batch
    correct = 0    # Sums the amount of correct predictions
    
    with torch.set_grad_enabled(is_train):
        for Bach_images, labels in loader:
            Bach_images, labels = Bach_images.to(device), labels.to(device)
            pred = model(Bach_images)
            loss = loss_fn(pred, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
    return total_loss / len(loader), correct / len(loader.dataset)