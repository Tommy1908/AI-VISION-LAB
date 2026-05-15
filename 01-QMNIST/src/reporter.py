import os
import textwrap
import wandb
import matplotlib.pyplot as plt

class ExperimentTracker:
    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.config = config
        self.history = {"train_loss": [], "test_loss": [], "test_acc": [], "gen_gap": []}
        self.samples = {"success": [], "failure": []}
        self.training_time = 0

        wandb.init(
            project=config["project_name"],
            name=exp_name,
            config=config,
            reinit=True,
            mode="offline" # Manual sync (wandb sync wandb/offline-run-*)
        )

    def log_epoch(self, train_loss, test_loss, test_acc):
        gap = test_loss - train_loss
        
        self.history["train_loss"].append(train_loss)
        self.history["test_loss"].append(test_loss)
        self.history["test_acc"].append(test_acc)
        self.history["gen_gap"].append(gap)

        wandb.log({
            "train/loss": train_loss,
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "generalization_gap": gap
        })

    def log_samples(self, img, label, probs,pred, is_success):
        category = "success" if is_success else "failure"
        if len(self.samples[category]) < 2:
            self.samples[category].append({
                "img": img, 
                "label": label, 
                "probs": probs,
                "pred": pred, 
            })

    def log_training_time(self, seconds):
        self.training_time += seconds
    
    def __format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)

        parts = []
        if h > 0: parts.append(f"{h}h")
        if m > 0 or h > 0: parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)

    def save_experiment_reports(self):
        os.makedirs(f"reports/experiments/{self.config['project_name']}", exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, ax = plt.subplots(2,2, figsize=(20,10))
        fig.suptitle(f"Experiment Report: {self.exp_name}", fontsize=20, fontweight='bold', y=0.95)

        #Train/Test Loss
        ax[0][0].plot(epochs, self.history["train_loss"], label="Train", color="blue")
        ax[0][0].plot(epochs, self.history["test_loss"], label="Test", color="red")
        ax[0][0].fill_between(epochs, self.history["train_loss"], self.history["test_loss"], color='gray', alpha=0.3, label='Gap', interpolate=True)
        ax[0][0].set_title(f"Test/TrainLoss - {self.exp_name}")
        #ax[0][0].set_ylim(0, 1)
        ax[0][0].legend()
        ax[0][0].grid(linestyle='--', alpha=0.5)
        
        #Accuracy
        ax[0][1].plot(epochs, self.history["test_acc"], label="Accuracy", color="green")
        ax[0][1].set_title(f"Accuracy (correct/total) - {self.exp_name}")
        #ax[0][1].set_ylim(0, 1)
        ax[0][1].legend()
        ax[0][1].grid(linestyle='--', alpha=0.5)


        #Generalization Gap
        ax[1][0].plot(epochs, self.history["gen_gap"], label="Gap", color="purple")
        ax[1][0].set_title(f"Generalization Gap (test_loss - train_loss) - {self.exp_name}")
        ax[1][0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        gen_gap = self.history["gen_gap"]
        ax[1][0].fill_between(epochs, gen_gap, 0, where=[g > 0 for g in gen_gap], 
                              color='red', alpha=0.2, label="Overfitting", interpolate=True)
        ax[1][0].fill_between(epochs, gen_gap, 0, where=[g <= 0.0 for g in gen_gap], 
                              color='green', alpha=0.2, label="Underfitting / Dropout", interpolate=True)
        #ax[1][0].set_ylim(-0.5, 0.5)
        ax[1][0].legend()
        ax[1][0].grid(linestyle='--', alpha=0.5)


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
        y_pos -= 0.05
        ax[1][1].text(x_left, y_pos, f"Training Time: {self.__format_time(self.training_time)}", **body_font)
        y_pos -= 0.12
        
        ax[1][1].text(x_left, y_pos, "HYPERPARAMETERS", **header_font)
        y_pos -= 0.08
        
        for k, v in self.config.items():
            text_to_wrap = f"{k:<15}: {v}"

            wrapped_lines = textwrap.wrap(text_to_wrap, width=40, subsequent_indent=" " * 3)

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


        plt.savefig(
            f"reports/experiments/{self.config['project_name']}/{self.exp_name}.png",
            dpi=300,
            bbox_inches='tight',
            )
        print(f"Report for experiment '{self.exp_name}' saved at 'reports/experiments/{self.exp_name}.png'")
        plt.close()
        
        return

    def finish(self):
        self.save_experiment_reports()
        wandb.finish()
        return self.history

def plot_project_master_report(project_name, all_results):
    os.makedirs("reports/projects", exist_ok=True)
    fig, ax = plt.subplots(2,2, figsize=(20,10))
    fig.suptitle(f"Project Report: {project_name.upper()}", fontsize=20, fontweight='bold', y=0.95)

    # Best results tracking
    best_overall_acc = -1
    best_acc_exp = ""
        
    best_gen_val = float('inf')
    best_gen_exp = ""

    for exp_name, history in all_results.items():
        epochs = range(1, len(history["test_loss"]) + 1)
        
        ax[0][0].plot(epochs, history["test_loss"], label=exp_name)
        ax[0][1].plot(epochs, history["test_acc"], label=exp_name)
        ax[1][0].plot(epochs, history["gen_gap"], label=exp_name)

        # Accuracy
        current_acc = history["test_acc"][-1] #We care abouth the final result
        if current_acc > best_overall_acc:
            best_overall_acc = current_acc
            best_acc_exp = exp_name
                        
        # Generalization gap
        current_gap = history["gen_gap"][-1] # Final generalization gap
        if current_gap < best_gen_val:
            best_gen_val = current_gap
            best_gen_exp = exp_name

    # Test loss
    ax[0][0].set_title(f"Test Loss")
    ax[0][0].set_ylim(0, 1)
    ax[0][0].legend()
    ax[0][0].grid(linestyle='--', alpha=0.5)
    
    #Accuracy
    ax[0][1].set_title(f"Accuracy")
    ax[0][1].set_ylim(0, 1)
    ax[0][1].legend()
    ax[0][1].grid(linestyle='--', alpha=0.5)

    #Generalization Gap
    ax[1][0].set_title(f"Generalization Gap")
    ax[1][0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax[1][0].set_ylim(-0.5, 0.5)
    ax[1][0].legend()
    ax[1][0].grid(linestyle='--', alpha=0.5)

    #Dashboard/overwiew
    ax[1][1].axis('off')

    ### Text font config
    header_font = {'fontsize': 13, 'weight': 'bold', 'family': 'sans-serif'}
    body_font = {'fontsize': 10, 'family': 'monospace', 'color': '#333333'}

    y_pos = 0.95
    x_pos = 0.05

    # Overall stats
    ax[1][1].text(x_pos, y_pos, f"PROJECT MASTER REPORT: {project_name.upper()}", **header_font)
    y_pos -= 0.08

    ax[1][1].text(x_pos, y_pos, "STATISTICS", **header_font)
    y_pos -= 0.08
    ax[1][1].text(x_pos, y_pos, f"Total Experiments: {len(all_results)}", **body_font)
    y_pos -= 0.05

    ax[1][1].text(x_pos, y_pos, "Best Accuracy:", **body_font)
    ax[1][1].text(x_pos + 0.2, y_pos, f"Acc: {best_overall_acc:.4f} ({best_acc_exp})", **body_font)
    y_pos -= 0.05

    ax[1][1].text(x_pos, y_pos, "Best Generalizer:", **body_font)
    ax[1][1].text(x_pos + 0.2, y_pos, f"Gap: {best_gen_val:.4f} ({best_gen_exp})", **body_font)
    y_pos -= 0.12
    
    plt.savefig(
        f"reports/projects/{project_name}.png",
        dpi=300,
        bbox_inches='tight',
        )
    print(f"Report for project '{project_name}' saved at 'reports/projects/{project_name}.png'")
    plt.close()
    return