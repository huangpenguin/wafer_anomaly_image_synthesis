import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_loss(save_dir):
    print("Starting analysis...")
    metrics_path = Path(save_dir) / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path)

    plt.figure(figsize=(10, 6))
    metrics_df.dropna(subset=["train_loss_epoch"]).plot(x="epoch", y="train_loss_epoch", legend=False)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss per Epoch")
    plt.title("Train Loss Curve")
    plt.grid(True)
    plt.savefig(Path(save_dir) / f"train_loss_curve.png") 
    print("Plotting loss curve...")
    plt.show(block=False)
    plt.pause(3) 
    plt.close()

if __name__ == "__main__":
    plot_training_loss("data\\output\\20241031\\rd4ad_resnet18_512_cc_25epoch\\version_0")