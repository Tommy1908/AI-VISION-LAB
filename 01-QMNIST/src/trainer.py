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

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss, _ = run_epoch(
                train_loader, self.model, self.loss_fn, self.optimizer, self.device
            )
            #Run with test to see each epoch performance, but without training
            test_loss, test_acc = run_epoch(
                test_loader, self.model, self.loss_fn, device=self.device, tracker=(self.tracker if epoch == self.config["epochs"] else None)
            )
            #tracker.log_epoch(epoch, train_loss, test_loss, test_acc)
            if self.tracker:
                self.tracker.log_epoch(train_loss, test_loss, test_acc)


            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} "
                  f"| Test Loss: {test_loss:.4f} | Acc: {100*test_acc:.1f}%")
        return


def run_epoch(loader, model, loss_fn, optimizer=None, device="cpu", tracker=None):
    is_train = optimizer is not None
    if is_train:
        model.train()  
    else:
        model.eval()
    total_loss = 0 # Sums the average loss of each batch
    correct = 0    # Sums the amount of correct predictions
    
    with torch.set_grad_enabled(is_train):
        for Batch_images, labels in loader:
            Batch_images, labels = Batch_images.to(device), labels.to(device)
            output = model(Batch_images)
            loss = loss_fn(output, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if tracker:
                all_probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                for i in range(len(labels)):
                    img = Batch_images[i].cpu()
                    label = labels[i].item()
                    probs = all_probs[i].tolist()
                    pred = preds[i].item() #Predicted class index
                    tracker.log_samples(img, label, probs, pred, is_success=(pred == label))

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == labels).type(torch.float).sum().item()
            
    return total_loss / len(loader), correct / len(loader.dataset)