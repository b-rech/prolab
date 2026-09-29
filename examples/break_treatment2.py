
# %% Dependencies

import prolab as pl
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter, test

# %% Open all sample measurements

raw = pl.read_files('data/TR_AQUARELA/all_samples',
                    instrument='perkinelmer',
                    format_string='{camp}_{pattern}_{id}_{rmode}_{msr}',
                    blank_pattern='REF',
                    sample_pattern='AMO',
                    depig_pattern='EXT',
                    decimal='.')

raw_melt = raw.melt(id_vars=['camp', 'pattern', 'id', 'rmode', 'msr', 'is_blank',
                             'is_sample', 'is_depig', 'is_total'],
                    var_name='wavelength',
                    value_name='absorbance')


grid = sns.FacetGrid(raw_melt, col='rmode', row='msr', hue='id',
                    margin_titles=True, despine=False, sharey=False,
                    height=1.5, aspect=2, palette='tab10')

grid.map(sns.lineplot, 'wavelength', 'absorbance', lw=.5)
sns.despine()

# %% Break treatment

# Breaks in the spectra
wl_central = (795, 685, 558, 422)
wl_critical = np.array([np.arange(wl + 2, wl - 3, -1) for wl in wl_central],
                       dtype=object)

uncorrected = raw.filter(regex='\d')
uncorrected_norm = uncorrected.div(uncorrected.mean(axis=1), axis=0)

# Calculate first and second derivatives
ddx = pd.DataFrame(np.gradient(uncorrected_norm, axis=1),
                   columns=uncorrected_norm.columns,
                   index=uncorrected_norm.index)
ddx2 = pd.DataFrame(np.gradient(ddx, axis=1),
                    columns=uncorrected_norm.columns,
                    index=uncorrected_norm.index)

# List to store corrected spectra
corrected_list = []

# Iterate over each spectrum
for i in uncorrected.index:

    # Get reference peak heights
    ddx_height = ddx.loc[i].abs().quantile(.95)
    ddx2_height = ddx2.loc[i].abs().quantile(.95)

    # Retrieve peaks
    ddx_peaks_idx = find_peaks(ddx.loc[i].abs(),
                               height=ddx_height, distance=10)
    dd2_peaks_idx = find_peaks(ddx2.loc[i].abs(),
                               height=ddx2_height, distance=10)
    ddx_peaks = uncorrected.columns[ddx_peaks_idx[0]]
    ddx2_peaks = uncorrected.columns[dd2_peaks_idx[0]]

    # Copy spectrum for correction
    spectrum_corr = uncorrected.loc[i].copy()

    # Check if peaks are in the critical wavelength ranges
    for wl_range in wl_critical:
        inter_ddx = set(wl_range).intersection(set(ddx_peaks))
        inter_ddx2 = set(wl_range).intersection(set(ddx2_peaks))

        title = 'No breaks detected'

        # Save critical range if peaks are found in both derivatives
        if len(inter_ddx) > 0 and len(inter_ddx2) > 0:

            title = 'Break detected'
            interval = spectrum_corr[wl_range].to_numpy()
            dist = interval[None, :] - interval[:, None]
            dist = pd.DataFrame(dist, index=wl_range, columns=wl_range)
            mask = np.triu(np.ones(dist.shape, dtype=bool))
            dist = dist.mask(~mask)

            wl_ref, wl_min = dist.abs().stack().idxmax()

            wl_to_corr = np.arange(wl_min, wl_ref+1, 1)

            offset = dist.stack()[(wl_ref, wl_min)]

            spectrum_corr.loc[:wl_min] -= offset

            #spectrum_corr.loc[wl_to_corr[1:-1]] = spectrum_corr.loc[wl_ref]

            for wlc in wl_to_corr:
                spectrum_corr.loc[wlc] = 2*spectrum_corr.loc[wl_ref] - spectrum_corr.loc[wl_ref+1]

    fig, ax1 = plt.subplots(figsize=(5.75, 5.75/2), dpi=300)
    ax2 = ax1.twinx()

    sns.lineplot(uncorrected.loc[i], lw=.5, color='red', ax=ax1)
    sns.scatterplot(uncorrected.loc[i], s=5, color='red', ax=ax1)
    sns.lineplot(spectrum_corr, lw=.5, color='green', ax=ax1)
    sns.scatterplot(spectrum_corr, s=5, color='green', ax=ax1)
    #sns.lineplot(ddx2.loc[i], lw=.5, color='blue', ax=ax2)
    ax1.set_title(title)

    plt.show()

    corrected_list.append(spectrum_corr)
# %%

# Breaks in the spectra
wl_central = (795, 685, 558, 422)
wl_critical = np.array([np.arange(wl + 2, wl - 3, -1) for wl in wl_central],
                       dtype=object)

for i in uncorrected.index:

    # Copy spectrum for correction
    spectrum_corr = uncorrected.loc[i].copy()

    lags = wavelength_differences(uncorrected_norm.loc[i])

    mean_lags = lags.mean(axis=1)
    ddx2_lags = pd.DataFrame({'ddx2': np.gradient(np.gradient(mean_lags))},
                             index=mean_lags.index)

    fig, ax1 = plt.subplots(figsize=(4, 3), dpi=200)
    ax2 = ax1.twinx()
    sns.lineplot(uncorrected.loc[i], lw=.5, color='red', ax=ax1)
    sns.lineplot(ddx2_lags, ax=ax2, lw=.5, color='blue', alpha=.5)
    plt.show()

    has_brake = input('Does it break? (y/n): ')

    if has_brake.lower() != 'y':
        continue

    else:
    
        critical = ddx2_lags.loc[wl_critical[-1]]
        wl_changes = np.array([critical.idxmin().item(), critical.idxmax().item()])
        wl_ref = 424
        wl_min = 421

        wl_to_corr = np.arange(wl_ref-1, wl_min-1, -1)

        value_min = spectrum_corr.loc[wl_min]

        for wlc in wl_to_corr:
            print(spectrum_corr.loc[wlc])
            print(spectrum_corr.loc[wlc+1])
            print(spectrum_corr.loc[wlc+2])
            print(2*spectrum_corr.loc[wlc+1] - spectrum_corr.loc[wlc+2])
            print('---')
            spectrum_corr.loc[wlc] = 2*spectrum_corr.loc[wlc+1] - spectrum_corr.loc[wlc+2:wlc+15].mean()   

        spectrum_corr.loc[:wl_min-1] += spectrum_corr.loc[wl_min] - value_min

        fig, ax1 = plt.subplots(figsize=(4, 3), dpi=500)
        sns.lineplot(uncorrected.loc[i], lw=.5, color='red', ax=ax1)
        sns.lineplot(spectrum_corr, lw=.5, color='green', ax=ax1)
        #plt.ylim(.195, .201)
        #plt.xlim(416, 430)
        plt.show()



#%%


def wavelength_differences(spectrum, min_lag=-10, max_lag=10):
    """Calculate spectral differences for wavelength lags in nanometers."""
    spectrum = spectrum.dropna().copy()
    spectrum.index = spectrum.index.astype(float)
    spectrum = spectrum.sort_index()

    wavelengths = spectrum.index.to_numpy()
    values = spectrum.to_numpy()

    differences = {}

    for lag in range(min_lag, max_lag + 1):
        shifted_wavelengths = wavelengths + lag

        shifted_values = np.interp(
            shifted_wavelengths,
            wavelengths,
            values,
            left=np.nan,
            right=np.nan
        )

        differences[lag] = values - shifted_values

    return pd.DataFrame(
        differences,
        index=wavelengths
    )

# %%

class InteractiveSpectrumEditor:
    """
    Controls:
    - e: toggle editing mode on/off.
    - Ctrl + left-click: add/remove a point from the selection.
    - Left-drag: move the selected points together.
    - Left-drag an unselected point: move that point only.
    - Right-drag: apply a vertical offset to the complete spectrum.
    - Shift + left-drag: offset points to the left of the selected point.
    - r: reset changes.
    - Enter: accept and close.
    - Escape: cancel and close.
    """

    def __init__(self, spectrum, reference=None):
        self.original = spectrum.copy().sort_index()
        self.edited = self.original.copy()
        self.reference = reference

        self.x = self.edited.index.to_numpy(dtype=float)
        self.initial_values = self.edited.to_numpy(dtype=float)

        self.editing_enabled = True
        self.selected_indices = set()

        self.active_index = None
        self.press_y = None
        self.press_values = None
        self.mode = None

        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=150)

        if reference is not None:
            self.ax.plot(
                reference.index,
                reference.values,
                color="red",
                lw=0.8,
                alpha=0.6,
                label="Original"
            )

        self.line, = self.ax.plot(
            self.x,
            self.edited.values,
            color="green",
            lw=1,
            label="Editable"
        )

        self.selected_points, = self.ax.plot(
            [],
            [],
            "o",
            color="black",
            ms=6,
            label="Selected"
        )

        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Absorbance")
        self.ax.legend()
        self.ax.grid(alpha=0.25)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def nearest_index(self, x):
        return np.abs(self.x - x).argmin()

    def update_plot(self):
        self.line.set_ydata(self.edited.to_numpy())

        if self.selected_indices:
            indices = sorted(self.selected_indices)
            self.selected_points.set_data(
                self.x[indices],
                self.edited.iloc[indices].to_numpy()
            )
        else:
            self.selected_points.set_data([], [])

        self.ax.set_title(
            f"Editing: {'ON' if self.editing_enabled else 'OFF'} | "
            f"Selected points: {len(self.selected_indices)}"
        )

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if not self.editing_enabled:
            return

        if event.inaxes != self.ax or event.xdata is None:
            return

        self.active_index = self.nearest_index(event.xdata)

        # Ctrl-click toggles selection without moving points.
        if event.button == 1 and event.key in ("control", "ctrl"):
            if self.active_index in self.selected_indices:
                self.selected_indices.remove(self.active_index)
            else:
                self.selected_indices.add(self.active_index)

            self.update_plot()
            return

        self.press_y = event.ydata
        self.press_values = self.edited.to_numpy().copy()

        if event.button == 3:
            self.mode = "global_offset"

        elif event.button == 1 and event.key == "shift":
            self.mode = "left_offset"

        elif event.button == 1:
            # Dragging a selected point moves the complete selection.
            if self.active_index in self.selected_indices:
                self.mode = "move_selected"
            else:
                self.selected_indices = {self.active_index}
                self.mode = "move_selected"

        self.update_plot()

    def on_motion(self, event):
        if (
            not self.editing_enabled
            or self.mode is None
            or event.inaxes != self.ax
            or event.ydata is None
        ):
            return

        delta = event.ydata - self.press_y

        if self.mode == "move_selected":
            indices = sorted(self.selected_indices)
            self.edited.iloc[indices] = (
                self.press_values[indices] + delta
            )

        elif self.mode == "global_offset":
            self.edited.iloc[:] = self.press_values + delta

        elif self.mode == "left_offset":
            self.edited.iloc[:self.active_index + 1] = (
                self.press_values[:self.active_index + 1] + delta
            )

        self.update_plot()

    def on_release(self, event):
        self.mode = None
        self.press_y = None
        self.press_values = None

    def on_key(self, event):
        if event.key == "e":
            self.editing_enabled = not self.editing_enabled
            self.mode = None
            self.update_plot()

        elif event.key == "r":
            self.edited.iloc[:] = self.initial_values
            self.selected_indices.clear()
            self.update_plot()

        elif event.key in ("enter", "return"):
            plt.close(self.fig)

        elif event.key == "escape":
            self.edited = self.original.copy()
            plt.close(self.fig)

    def show(self):
        self.update_plot()
        plt.show()
        return self.edited.copy()


def edit_spectrum_interactively(spectrum, reference=None):
    editor = InteractiveSpectrumEditor(
        spectrum=spectrum,
        reference=reference
    )
    return editor.show()



spectrum_corr = edit_spectrum_interactively(
    spectrum=uncorrected.loc[i],
    reference=None)


    # Continue using spectrum_corr here

# ...existing code...
# %%
