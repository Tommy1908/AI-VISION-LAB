import os
import textwrap
import matplotlib.pyplot as plt

class ExperimentTracker:
    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.config = config
        self.history = {"train_loss": [], "test_loss": [], "test_acc": [], "gen_gap": []}
        self.samples = {"success": [], "failure": []}

    def log_epoch(self, train_loss, test_loss, test_acc):
        gap = test_loss - train_loss
        
        self.history["train_loss"].append(train_loss)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["gen_gap"].append(gap)

    def log_samples(self, img, label, probs,pred, is_success):
        category = "success" if is_success else "failure"
        if len(self.samples[category]) < 2:
            self.samples[category].append({
                "img": img, 
                "label": label, 
                "probs": probs,
                "pred": pred, 
            })

    def save_experiment_reports(self):
        os.makedirs("reports/experiments", exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, ax = plt.subplots(2,2, figsize=(20,10))

        #Train/Test Loss
        ax[0][0].plot(epochs, self.history["train_loss"], label="Train", color="blue")
        ax[0][0].plot(epochs, self.history["test_loss"], label="Test", color="red")
        ax[0][0].fill_between(epochs, self.history["train_loss"], self.history["test_loss"], color='gray', alpha=0.3, label='Gap', interpolate=True)
        ax[0][0].set_title(f"Test/TrainLoss - {self.exp_name}")
        ax[0][0].set_ylim(0, 0.8)
        ax[0][0].legend()

        
        #Accuracy
        ax[0][1].plot(epochs, self.history["test_acc"], label="Accuracy", color="green")
        ax[0][1].set_title(f"Accuracy (correct/total) - {self.exp_name}")
        ax[0][1].set_ylim(0.2, 1)
        ax[0][1].legend()

        #Generalization Gap
        ax[1][0].plot(epochs, self.history["gen_gap"], label="Gap", color="purple")
        ax[1][0].set_title(f"Generalization Gap (test_loss - train_loss) - {self.exp_name}")
        ax[1][0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        gen_gap = self.history["gen_gap"]
        ax[1][0].fill_between(epochs, gen_gap, 0, where=[g > 0 for g in gen_gap], 
                              color='red', alpha=0.2, label="Overfitting", interpolate=True)
        ax[1][0].fill_between(epochs, gen_gap, 0, where=[g <= 0.0 for g in gen_gap], 
                              color='green', alpha=0.2, label="Underfitting / Dropout", interpolate=True)
        ax[1][0].set_ylim(-0.5, 0.5)
        ax[1][0].legend()

        #Dashboard/overwiew
        ax[1][1].axis('off')

        ### Calculamos las mejores métricas
        best_acc = max(self.history["test_acc"])
        best_epoch = self.history["test_acc"].index(best_acc) + 1
        min_test_loss = min(self.history["test_loss"])
        
        ### Text font config
        header_font = {'fontsize': 13, 'weight': 'bold', 'family': 'sans-serif'}
        body_font = {'fontsize': 10, 'family': 'monospace', 'color': '#333333'}


        ### Left Column ###
        y_pos = 0.95
        x_left = 0.05

        ax[1][1].text(x_left, y_pos, "RESULTS SUMMARY", **header_font)
        y_pos -= 0.08
        ax[1][1].text(x_left, y_pos, f"Best Accuracy: {best_acc:.4f} (Ep {best_epoch})", **body_font)
        y_pos -= 0.05
        ax[1][1].text(x_left, y_pos, f"Min Test Loss: {min_test_loss:.4f}", **body_font)
        y_pos -= 0.12 # Salto de sección
        
        ax[1][1].text(x_left, y_pos, "HYPERPARAMETERS", **header_font)
        y_pos -= 0.08
        
        for k, v in self.config.items():
            text_to_wrap = f"{k:<15}: {v}"

            wrapped_lines = textwrap.wrap(text_to_wrap, width=50, subsequent_indent=" " * 5)

            for line in wrapped_lines:
                ax[1][1].text(x_left, y_pos, line, **body_font)
                y_pos -= 0.05


        ### Right Column ###

        y_pos = 0.95
        x_right = 0.55
        ax[1][1].text(x_right, y_pos, "PREDICTION SAMPLES", **header_font)
        
        #Some hardcoded values to position the samples (current display was 20,10, so its widther than taller)
        width_img = 0.125
        height_img = 0.25
        x_offset = 0.22
        
        # Success
        y_ok = 0.55
        for i, sample in enumerate(self.samples["success"]):
            ax_ok = ax[1][1].inset_axes([x_right + (i * x_offset), y_ok, width_img, height_img])
            ax_ok.imshow(sample["img"].squeeze(), cmap='gray')
            #Probs
            prob_pred = sample["probs"][sample["pred"]]
            ok_label = f"OK: {sample['label']}\nProb: {prob_pred:.2f}"
            ax_ok.set_title(ok_label, fontsize=10, color='green')
            ax_ok.axis('off')

        ## Failure
        y_fail = 0.15
        for i, sample in enumerate(self.samples["failure"]):
            ax_fail = ax[1][1].inset_axes([x_right + (i * x_offset), y_fail, width_img, height_img])
            ax_fail.imshow(sample["img"].squeeze(), cmap='gray')
            #Probs
            prob_pred = sample["probs"][sample["pred"]]
            prob_actual = sample["probs"][sample["label"]]
            fail_label = f"PRED: {sample['pred']} ({prob_pred:.2f})\nActual: {sample['label']} ({prob_actual:.2f})"
            ax_fail.set_title(fail_label, fontsize=10, color='red')
            ax_fail.axis('off')


        plt.savefig(f"reports/experiments/{self.exp_name}.png")
        plt.close()
        
        return

    def finish(self):
        self.save_experiment_reports()
        return self.history

def plot_project_master_report(project_name, all_results):
    os.makedirs("reports/projects", exist_ok=True)

    fig, ax = plt.subplots(2,2, figsize=(20,10))
    for exp_name, history in all_results.items():
        epochs = range(1, len(history["test_loss"]) + 1)
        
        ax[0][0].plot(epochs, history["test_loss"], label=exp_name)
        ax[0][1].plot(epochs, history["test_acc"], label=exp_name)
        ax[1][0].plot(epochs, history["gen_gap"], label=exp_name)

    # Test loss
    ax[0][0].set_title(f"Test Loss")
    ax[0][0].set_ylim(0, 0.8)
    ax[0][0].legend()
    
    #Accuracy
    ax[0][1].set_title(f"Accuracy")
    ax[0][1].set_ylim(0.2, 1)
    ax[0][1].legend()

    #Generalization Gap

    ax[1][0].set_title(f"Generalization Gap")
    ax[1][0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax[1][0].set_ylim(-0.5, 0.5)
    ax[1][0].legend()

    plt.savefig(f"reports/projects/{project_name}.png")
    plt.close()
    return