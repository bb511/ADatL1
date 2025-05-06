import numpy as np
import h5py
import pandas as pd


def get_threshold_dict(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        # Navigate to the "results/threshold" group
        thresholds_group = h5_file["results/threshold"]

        # Extract the keys and values datasets
        keys = np.array(thresholds_group["key"])
        values = np.array(thresholds_group["value"])

        # Create the dictionary
        threshold_dict = {
            key.decode("utf-8"): value for key, value in zip(keys, values)
        }

    return threshold_dict


def get_history_dict(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        # Navigate to the "history" group
        history_group = h5_file["history"]

        # Create the dictionary to hold the history data
        history_dict = {}

        # Loop through each dataset in the history group
        for key in history_group.keys():
            # Extract the dataset and store it in the dictionary
            history_dict[key] = np.array(history_group[key])

    return history_dict


def get_axo_score_dataframes(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        # Navigate to the "axo_scores" group inside the "results" group
        axo_scores_group = h5_file["results/axo_scores"]

        # Create the dictionary to hold the DataFrames
        axo_score_dfs = {}

        # Loop through each threshold group
        for thres in axo_scores_group.keys():
            # Retrieve the group for the specific threshold
            thres_group = axo_scores_group[thres]

            # Create a dictionary to hold the data for the DataFrame
            data_dict = {}

            # Loop through each dataset in the threshold group
            for key in thres_group.keys():
                data = np.array(thres_group[key])

                # Decode string columns from bytes to strings
                if "name" in key.lower():  # Check if the data is of byte string type
                    data_dict[key] = [item.decode("utf-8") for item in data]
                else:
                    data_dict[key] = data

            # Convert the dictionary to a pandas DataFrame
            df = pd.DataFrame(data_dict)

            # Reorder the columns as required
            desired_order = [
                "Signal Name",
                "AXO SCORE",
                "L1 SCORE",
                "HT SCORE",
                "AXO Improvement",
            ]
            df = df[
                desired_order + [col for col in df.columns if col not in desired_order]
            ]

            # Store the DataFrame in the dictionary with the threshold as the key
            axo_score_dfs[thres] = df

    return axo_score_dfs


def get_histogram_dict(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        # Create the dictionary to hold the histograms
        hist_dict = {}

        # Access the background histograms
        background_group = h5_file["results/score_hist/background"]
        background_count = np.array(background_group["count"])
        background_bin = np.array(background_group["bin"])
        hist_dict["background"] = (background_count, background_bin)

        # Access the signal histograms
        signal_group = h5_file["results/score_hist/signal"]

        # Loop through each signal group
        for signal_name in signal_group.keys():
            signal_hist_group = signal_group[signal_name]

            # Retrieve the count and bin datasets
            signal_count = np.array(signal_hist_group["count"])
            signal_bin = np.array(signal_hist_group["bin"])

            # Store the tuple (count, bin_edges) in the dictionary under the signal name
            hist_dict[signal_name] = (signal_count, signal_bin)

    return hist_dict
