import click
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv


def read_single_features(path: str) -> list[str]:
    with open(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        lines = list(reader)

    return [k[0] for k in lines[1:] if float(k[1]) == 1.0]


def generate_plot(hits, single_features, output_path):
    train_data = np.log2(hits.drop(["label"], axis=1) + 1)
    number_of_samples, number_of_features = train_data.shape

    row_groups = pd.Series(hits["label"] == "other", index=hits.index)
    row_palette = {False: "lightcoral", True: "lightgray"}
    row_colors = row_groups.map(row_palette)

    if single_features is not None:
        col_groups = pd.Series(
            [k in single_features for k in hits.columns], index=hits.columns
        )
        col_palette = {False: "white", True: "c"}
        col_colors = col_groups.map(col_palette)
    else:
        col_colors = None

    cm = sns.clustermap(
        train_data,
        cmap="Blues",
        row_colors=row_colors,
        col_colors=col_colors,
        col_cluster=True,
        yticklabels=True,
    )
    plt.setp(
        cm.ax_heatmap.yaxis.get_majorticklabels(),
        fontsize=150 / number_of_samples,
        rotation=0,
    )
    cm.ax_col_dendrogram.set_title(
        f"A heat-map of the significance of the top {number_of_features} discriminatory motifs"
    )
    cm.savefig(f"{output_path}.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close()


@click.command()
@click.argument("hits_path")
@click.argument("output_path")
@click.option("--single_feature_path")
def generate_heatmap(
    hits_path: str, output_path: str, single_feature_path: str | None
) -> None:
    hits = pd.read_csv(hits_path, index_col="sample_name")
    if single_feature_path is not None:
        single_features = read_single_features(single_feature_path)
    else:
        single_features = None
    generate_plot(hits, single_features, output_path)


if __name__ == "__main__":
    generate_heatmap()
