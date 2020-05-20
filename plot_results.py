import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
from utils import make_dirs


def extract_3_metrics(met_fname):

    met_df = pd.read_csv(met_fname, index_col=0)
    met_df = met_df[["pair", "accuracy", "MAP", "precision@10"]]

    ndf = pd.melt(met_df, id_vars="pair",
                  value_vars=["accuracy", "MAP", "precision@10"],
                  var_name='Metric', value_name='Value')

    ndf['model'] = os.path.basename(os.path.dirname(met_fname)).split("_")[0]

    return ndf


def extract_precision_metrics(met_fname):

    met_df = pd.read_csv(
        met_fname,
        index_col=0)

    met_df = met_df[["pair", "precision@1", "precision@3", "precision@5", "precision@10", "precision@30"]]

    ndf = pd.melt(met_df, id_vars="pair",
                  value_vars=["precision@1", "precision@3", "precision@5", "precision@10", "precision@30"],
                  var_name='k', value_name='Precision@k')

    ndf['k'] = ndf['k'].map({
        "precision@1": 1,
        "precision@3": 3,
        "precision@5": 5,
        "precision@10": 10,
        "precision@30": 30
    })

    ndf['model'] = os.path.basename(os.path.dirname(met_fname)).split("_")[0]

    return ndf


def generate_precision_plots(prec_df, plot_outdir):

    sns.catplot(x="k", y="Precision@k", hue="model", hue_order=['AnchoredSiameseGNN'], #, 'SiameseGNN'],
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=prec_df)

    make_dirs(plot_outdir)
    plt.savefig(os.path.join(plot_outdir, "precision.eps"))
    plt.close()


def generate_3_metrics_plots(metrics_df, x_label, plot_outdir):

    sns.catplot(x=x_label, y="Value", hue="model", hue_order=['AnchoredSiameseGNN'], #, 'SiameseGNN'],
                col="Metric", capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=metrics_df)

    make_dirs(plot_outdir)
    plt.savefig(os.path.join(plot_outdir, "3_metrics_plot.eps"))
    plt.close()


def compare_models_on_arXiv(met_dir="metrics/arXiv"):

    prec_df = [extract_precision_metrics(f)
               for f in glob.glob(os.path.join(met_dir, "*/paired_metrics.csv"))]

    prec_df = pd.concat(prec_df)

    generate_precision_plots(prec_df, "plots/arXiv")


def compare_models_on_synthetic(synthetic_glob_dir, var_name):

    score_dfs = []

    star_pos = 0

    for i, s in enumerate(synthetic_glob_dir.split("/")):
        if s == "*":
            star_pos = i
            break

    for met_f in glob.glob(synthetic_glob_dir):
        df = extract_3_metrics(met_f)

        if var_name == "Test ratio":
            var_val = met_f.split("tr_")[1].split("_")[0]
        elif var_name == "Random graph type":
            var_val = met_f.split("/")[star_pos]
        else:
            var_val = met_f.split("/")[star_pos].split("_")[1]

        try:
            var_val = float(var_val)
        except Exception:
            pass

        df[var_name] = var_val

        score_dfs.append(df)

    score_dfs = pd.concat(score_dfs)
    generate_3_metrics_plots(score_dfs, var_name, os.path.join("plots/synthetic/", var_name))



if __name__ == '__main__':

    compare_models_on_arXiv()

    compare_models_on_synthetic("metrics/data/synthetic/PA/n_100/erp_0.1/*/AnchoredSiameseGNN*_tr_0.8*/paired_metrics.csv",
                                "Node removal probability")

    compare_models_on_synthetic("metrics/data/synthetic/PA/n_100/*/nrp_0.1/AnchoredSiameseGNN*_tr_0.8*/paired_metrics.csv",
                                "Edge removal probability")

    compare_models_on_synthetic("metrics/data/synthetic/PA/n_100/erp_0.1/nrp_0.1/AnchoredSiameseGNN*/paired_metrics.csv",
                                "Test ratio")

    compare_models_on_synthetic("metrics/data/synthetic/*/n_100/erp_0.1/nrp_0.1/AnchoredSiameseGNN*_tr_0.8*/paired_metrics.csv",
                                "Random graph type")

    compare_models_on_synthetic("metrics/data/synthetic/PA/*/erp_0.1/nrp_0.1/AnchoredSiameseGNN*_tr_0.8*/paired_metrics.csv",
                                "Graph size")