#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:42:54 2025

@author: rachelwillis
"""

import matplotlib.pyplot as plt
import numpy as np

# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})


def plotChRange(data, time_index, startCh=0, endCh=None, fs=1000, clipPerc=None,
                time_axis='vertical', cmap='seismic', dpi=200, title=None,
                outfile=None):
    """
    Plot DAS profile for given channel range

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered,  in case only part of the channels is
        of interest.
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """

    if time_axis != 'vertical':
        data = data.T

    if not endCh:
        endCh = data.shape[1]

    if clipPerc:
        clip = np.percentile(np.absolute(data), clipPerc)
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-clip, vmax=clip, cmap=cmap)#, extent=(startCh, endCh-1,init_t, fin_t))
    else:
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-abs(data).max(), vmax=abs(data).max(),
                   cmap=cmap)#, extent=(startCh, endCh-1, init_t, fin_t))
    plt.xlabel('Channel number')
    plt.xticks(np.linspace(0, 500, 6).tolist(), np.linspace(startCh, startCh + 500, 6).tolist())
    plt.ylabel('Time (s)')
    plt.yticks(np.linspace(0, 660, 4).tolist(), np.linspace(time_index/220, (time_index+660)/220, 4).tolist())

    plt.colorbar(label='Strain rate (10$^{-9}$ s$^{-1}$)')
    plt.tight_layout()
    if title:
        plt.title(title, fontsize=16)
    if outfile:
        plt.savefig(outfile, format='png')
        plt.close('all')
        
    return


def plot_covariance_and_features(coh_matrix_ave, features, startChLoc, starttime, time_index, save=True):
    """
    Plots the covariance matrix and associated feature line plots (Max Eigenvalue,
    Coherency, and Variance) in a combined figure.

    Parameters
    ----------
    coh_matrix_ave : np.ndarray
        3D array representing the covariance matrices (e.g., shape [time, channels, channels]).
    features : np.ndarray
        2D array of extracted features with columns [Max Eigenvalue, Coherency, Variance].
    startChLoc : int
        Starting channel number for labeling.
    starttime : str
        Start time string used in the figure title and save file name.
    time_index : int
        Time index corresponding to the covariance matrix.
    save : bool, optional
        If True, saves the plot to a file. Default is True.
    """

    fig = plt.figure(figsize=(18, 8))

    # --- Covariance Matrix ---
    ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=2)
    im = ax1.imshow(np.abs(coh_matrix_ave[30, :, :]), cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax1, label="Covariance Magnitude")
    ax1.set_title("Covariance Matrix")
    ax1.set_xlabel("Channel number")
    ax1.set_xticks(np.linspace(-0.5, 124.5, 6).tolist())
    ax1.set_xticklabels(np.linspace(startChLoc, startChLoc + 500, 6).astype(int))
    ax1.set_ylabel("Channel number")
    ax1.set_yticks(np.linspace(-0.5, 124.5, 6).tolist())
    ax1.set_yticklabels(np.linspace(startChLoc, startChLoc + 500, 6).astype(int))

    # --- Line Plots (Right Column) ---
    ax2 = plt.subplot2grid((6, 4), (0, 2), colspan=2, rowspan=2)  # Max Eigenvalue
    ax3 = plt.subplot2grid((6, 4), (2, 2), colspan=2, rowspan=2, sharex=ax2)  # Coherency
    ax4 = plt.subplot2grid((6, 4), (4, 2), colspan=2, rowspan=2, sharex=ax2)  # Variance

    # Max Eigenvalue
    ax2.plot(features[:, 0], label="Max Eigenvalue", color="blue")
    ax2.set_title("Covariance Metrics vs Frequency")
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc='upper right', fontsize=20)

    # Coherency
    ax3.plot(features[:, 1], label="Coherency", color="orange")
    ax3.tick_params(labelbottom=False)
    ax3.legend(loc='upper right', fontsize=20)

    # Variance
    ax4.plot(features[:, 2], label="Variance", color="green")
    ax4.set_xlabel("Frequency Bins")
    ax4.legend(loc='upper right', fontsize=20)

    # Adjust layout
    plt.tight_layout(pad=0.1)

    # Save or show
    if save:
        save_path = f'{starttime}_ch_{startChLoc}_to_{startChLoc + 500}_tind_{time_index}_to_{time_index+3}_combined_shared_x.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)

