import torch
import torch.nn as nn
from configs.experiments import EXPERIMENTS
from src.data import get_dataloaders
from src.model import QMNIST_MLP
from src.trainer import Trainer
from src.reporter import ExperimentTracker, plot_project_master_report

def run_experiment(exp_name, config):
    print(f"Running experiment: {exp_name}")
    train_loader, test_loader = get_dataloaders()
    model:QMNIST_MLP = QMNIST_MLP(hidden_layers=config["hidden_layers"], activation=config["activation"]).to(config['device'])
    print(f"Model created:\n{model}")
    
    tracker = ExperimentTracker(exp_name, config)
    Trainer(model,config, tracker).fit(train_loader, test_loader)

    return tracker.finish()

if __name__ == "__main__":
    # Structure: { "project_name": { "exp_1": history, "exp_2": history } }
    results_by_project = {}

    for exp_name, config in EXPERIMENTS.items():
        history = run_experiment(exp_name, config)
        
        project_name = config.get("project_name", "NewProject")
        if project_name not in results_by_project:
            results_by_project[project_name] = {}
        
        results_by_project[project_name][exp_name] = history

    # Generate reports for each project
    for project_name, experiments_data in results_by_project.items():
        plot_project_master_report(project_name, experiments_data)