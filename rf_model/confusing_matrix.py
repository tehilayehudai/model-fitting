import click
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def confusing_matrix(prediction_path):
    data = pd.read_csv(prediction_path, delimiter="\t", skiprows=1)
    true_labels = data["true label"]
    predicted_labels = data["prediction"]
    labels = true_labels.unique().tolist()
    confusion_mat = confusion_matrix(
        true_labels,
        predicted_labels,
    )
    return confusion_mat, labels


def generate_plot(output_path, confusion_mat, labels, num):
    plt.figure()
    sns.heatmap(
        confusion_mat,
        annot=True,
        cmap="Blues",
        cbar=False,
        xticklabels=(labels),
        yticklabels=(labels),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Number of motifs: {num}")
    plt.savefig(output_path)


@click.command()
@click.argument("prediction_path", type=str)
@click.argument("output_path", type=str)
def generate_confusing_matrix(prediction_path, output_path):
    num = (os.path.basename(prediction_path).split("_"))[1]
    output_path_png = f"{output_path}.png"
    confusion_mat, labels = confusing_matrix(prediction_path)
    generate_plot(output_path_png, confusion_mat, labels, num)


if __name__ == "__main__":
    generate_confusing_matrix()
