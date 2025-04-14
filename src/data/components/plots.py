# Plotting of the features that are stored in the converted h5s.

import os
import operator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def plot_hist_3d(data: np.ndarray, object_type: str, feats: list, outdir: Path):
    """Plots data in 3d numpy array.

    The expected format of the data is nevents x nobjects x feats.
    Thus, this method plots a histogram of each feature of each object across all
    events in the given data numpy array.
    """
    outdir = outdir / f"{object_type}_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    for feat_nb in range(data.shape[2]):
        # Don't plot features that are not specified in the given feature list.
        if feat_nb >= len(feats):
            continue

        plt.xlim(np.nanmin(data[:, :, feat_nb]), np.nanmax(data[:, :, feat_nb]))
        plt.figure(figsize=(12, 10))
        counts, edges, bars = plt.hist(
            x=data[:, :, feat_nb].flatten(),
            bins=30,
            alpha=0.5,
            linewidth=1.5,
            color="lavender",
            edgecolor="midnightblue",
        )

        plt.xlabel(feats[feat_nb])
        plt.ylabel("Counts")
        plt.gca().set_yscale("log")
        plt.text(
            0.68,
            0.95,
            f"Total Counts: {int(sum(counts))}",
            transform=plt.gca().transAxes,
            fontsize=15,
        )

        plt.savefig(outdir / f"{feats[feat_nb]}.png")
        plt.close()

    log.info(f"Plots for {object_type} saved to: {outdir}")


def plot_hist_2d(data: np.ndarray, object_type: str, feats: list, outdir: Path):
    """Plots data in 2d numpy array.

    The expected format of the data is nevents x feats.
    Thus, this method plots a histogram of each feature across all events.
    """
    outdir = outdir / f"{object_type}_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

    for feat_nb in range(data.shape[1]):
        plt.xlim(np.nanmin(data[:, feat_nb]), np.nanmax(data[:, feat_nb]))
        plt.figure(figsize=(12, 10))
        counts, edges, bars = plt.hist(
            x=data[:, feat_nb].flatten(),
            bins=30,
            alpha=0.5,
            linewidth=1.5,
            color="lavender",
            edgecolor="midnightblue",
        )

        plt.xlabel(feats[feat_nb])
        plt.ylabel("Counts")
        plt.gca().set_yscale("log")
        plt.text(
            0.68,
            0.95,
            f"Total Counts: {int(sum(counts))}",
            transform=plt.gca().transAxes,
            fontsize=15,
        )
        plt.savefig(outdir / f"{feats[feat_nb]}.png")
        plt.close()

    log.info(f"Plots for {object_type} saved to: {outdir}")


def plot_hist_1d(data: np.ndarray, feat: str, outdir: Path):
    """Plots data in 1d numpy array."""
    outdir = outdir / f"{feat}_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

    if not (np.array(data) > 0.1).any():
        log.info("Array is full of 0s, skipping the plotting!")
        return

    plt.figure(figsize=(12, 10))
    plt.xlim(np.nanmin(data), np.nanmax(data))
    counts, edges, bars = plt.hist(
        x=np.array(data).flatten(),
        bins=30,
        alpha=0.5,
        linewidth=1.5,
        color="lavender",
        edgecolor="midnightblue",
    )
    plt.xlabel(feat)
    plt.ylabel("Counts")
    plt.gca().set_yscale("log")
    plt.text(
        0.68,
        0.95,
        f"Total Counts: {int(sum(counts))}",
        transform=plt.gca().transAxes,
        fontsize=15,
    )
    plt.savefig(outdir / f"{feat}.png")
    plt.close()

    log.info(f"Plots for {feat} saved to: {outdir}")
