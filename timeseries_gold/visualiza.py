import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, dataset_csv, real_csv, predict_csv, n_history=200000):
        self.dataset_csv = dataset_csv
        self.real_csv = real_csv
        self.predict_csv = predict_csv
        self.n_history = n_history

    def plot_predictions_all(self):
        """
        Plot Open, High, Low, Close from dataset, real, and predictions.
        Each target is shown in its own subplot.
        """
        # Load data
        df_hist = pd.read_csv(self.dataset_csv, parse_dates=["Time"])
        df_real = pd.read_csv(self.real_csv, parse_dates=["Time"])
        df_pred = pd.read_csv(self.predict_csv, parse_dates=["Time"])

        # Slice last N history rows for context
        df_hist_tail = df_hist.tail(self.n_history)

        targets = ["Open", "High", "Low", "Close"]

        fig, axes = plt.subplots(len(targets), 1, figsize=(12, 10), sharex=True)

        for ax, target_col in zip(axes, targets):
            # Historical (training input)
            ax.plot(df_hist_tail["Time"], df_hist_tail[target_col],
                    label="History (dataset)", color="blue")

            # Real values after dataset ends
            ax.plot(df_real["Time"], df_real[target_col],
                    label="Real data", color="green")

            # Predictions
            ax.plot(df_pred["Time"], df_pred[target_col],
                    label="Predictions", color="red", linestyle="--")

            ax.set_ylabel(target_col)
            ax.legend(loc="best")
            ax.grid(True, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Predicted vs Real OHLC", fontsize=14)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        plt.show()
