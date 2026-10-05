"""Scientific plots and SHAP from executed predictions, never illustrative metrics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def make_figures(table, predictions, xgb_pipeline, test, result, reports):
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    annual = table.groupby("FYEAR").YIELD.agg(["mean", "std"])
    ax.plot(annual.index, annual["mean"], marker="o", color="#176b50")
    ax.fill_between(
        annual.index,
        annual["mean"] - annual["std"],
        annual["mean"] + annual["std"],
        color="#176b50",
        alpha=0.12,
        label="County ±1 SD (dispersion)",
    )
    ax.axvspan(2013.5, 2015.5, color="#c29231", alpha=0.12, label="Validation")
    ax.axvspan(2015.5, 2018.5, color="#4574a6", alpha=0.12, label="Final test")
    ax.set(
        xlabel="Year",
        ylabel="Maize yield (bushels/acre)",
        title="Retained counties: annual yield and temporal split",
    )
    ax.legend(loc="lower right")
    fig.savefig(figures / "yield-by-year.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(predictions.YIELD, predictions.prediction, s=9, alpha=0.25, color="#176b50")
    lo, hi = predictions.YIELD.min(), predictions.YIELD.max()
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
    ax.set(
        xlabel="Observed (bu/ac)",
        ylabel="Predicted (bu/ac)",
        title=f"Final test: {result['selected_model']}",
    )
    ax = axes[1]
    residual = predictions.prediction - predictions.YIELD
    ax.scatter(predictions.prediction, residual, s=9, alpha=0.25, color="#4574a6")
    ax.axhline(0, color="black", linestyle="--")
    ax.set(
        xlabel="Predicted (bu/ac)",
        ylabel="Prediction − observation (bu/ac)",
        title="Residuals reveal failure modes",
    )
    fig.tight_layout()
    fig.savefig(figures / "test-diagnostics.png")
    plt.close(fig)

    names = list(result["test"])
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = np.arange(len(names))
    ax.bar(
        pos - 0.18,
        [result["validation"][n]["rmse"] for n in names],
        0.36,
        label="Validation (train fit)",
        color="#c29231",
    )
    ax.bar(
        pos + 0.18,
        [result["test"][n]["rmse"] for n in names],
        0.36,
        label="Final test (train+val refit)",
        color="#176b50",
    )
    ax.set_xticks(pos, names)
    ax.set(
        ylabel="RMSE (bushels/acre), lower is better",
        title="Fixed candidates; selection uses validation only",
    )
    ax.legend()
    fig.savefig(figures / "model-comparison.png")
    plt.close(fig)

    grouped = pd.read_csv(reports / "group_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dimension in zip(axes, ["FYEAR", "STATE"], strict=True):
        values = grouped.loc[grouped.dimension == dimension]
        ax.bar(values.value.astype(str), values.rmse, color="#4574a6")
        ax.set(
            xlabel=dimension, ylabel="RMSE (bu/ac)", title=f"Selected model by {dimension.lower()}"
        )
    fig.tight_layout()
    fig.savefig(figures / "group-errors.png")
    plt.close(fig)

    sample = test.sample(n=min(300, len(test)), random_state=result["config"]["seed"])
    preprocess = xgb_pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(sample)
    names = preprocess.get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(xgb_pipeline.named_steps["model"])
    explanation = explainer(transformed)
    importance = np.abs(explanation.values).mean(axis=0)
    pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(reports / "shap_importance.csv", index=False)
    order = np.argsort(importance)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order], importance[order], color="#176b50")
    ax.set(
        xlabel="Mean |SHAP| (bu/ac)",
        title="XGBoost attribution · 300 final-test rows · descriptive",
    )
    fig.savefig(figures / "shap-importance.png")
    plt.close(fig)
    predicted = xgb_pipeline.predict(sample)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    if not np.allclose(predicted, reconstructed, atol=0.002, rtol=0.0001):
        raise ValueError("SHAP contributions do not reconstruct predictions")
