import numpy as np
import matplotlib.pyplot as plt
import os
from plot_utils import set_science_style, get_figure_size, get_color_cycle, HIGH_CONTRAST_COLORS, FONT_SIZES
from scipy.signal import find_peaks

spectrum_data_folderpath = "./lab5/spectrum_data"

def plot_spectrum(file_path, wavelength_min=None, wavelength_max=None, color_idx=0, 
                  title=None, label=None, show_plot=True, ax=None, 
                  find_absorption=False, window_size=15, min_drop_percent=1.0,
                  min_abs_drop=None, max_peaks=15, proximity_threshold=5,
                  marker_offset=0.02, marker_text_offset=0.07):

    wavelengths = []
    intensities = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        wavelength = float(parts[0])
                        intensity = float(parts[1])
                        wavelengths.append(wavelength)
                        intensities.append(intensity)
                    except ValueError:
                        continue
    
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)
    
    if wavelength_min is not None or wavelength_max is not None:
        mask = np.ones_like(wavelengths, dtype=bool)
        if wavelength_min is not None:
            mask = mask & (wavelengths >= wavelength_min)
        if wavelength_max is not None:
            mask = mask & (wavelengths <= wavelength_max)
        
        wavelengths = wavelengths[mask]
        intensities = intensities[mask]
    
    if ax is None:
        set_science_style(use_tex=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plt.rcParams.update({
            'font.size': FONT_SIZES['large'],
            'axes.labelsize': FONT_SIZES['xlarge'],
            'axes.titlesize': FONT_SIZES['xlarge'],
            'xtick.labelsize': FONT_SIZES['xlarge'],
            'ytick.labelsize': FONT_SIZES['xlarge'],
            'legend.fontsize': FONT_SIZES['xlarge']
        })
    else:
        fig = ax.figure
    
    color = HIGH_CONTRAST_COLORS[color_idx % len(HIGH_CONTRAST_COLORS)]
    
    ax.plot(wavelengths, intensities, color=color, linewidth=1.5, label=label)
    
    absorption_peaks = []
    if find_absorption:
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        
        rolling_avg = np.zeros_like(intensities)
        for i in range(len(intensities)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(intensities), i + half_window + 1)
            rolling_avg[i] = np.mean(intensities[start_idx:end_idx])
        
        all_minima = []
        for i in range(1, len(intensities) - 1):
            if (intensities[i] < intensities[i-1] and 
                intensities[i] < intensities[i+1]):
                
                abs_drop = rolling_avg[i] - intensities[i]
                rel_drop_percent = 100 * abs_drop / rolling_avg[i]
                
                all_minima.append({
                    'index': i,
                    'wavelength': wavelengths[i],
                    'intensity': intensities[i],
                    'background': rolling_avg[i],
                    'abs_drop': abs_drop,
                    'drop_percent': rel_drop_percent
                })
        
        potential_peaks = []
        for peak in all_minima:
            meets_criteria = True
            
            if peak['drop_percent'] < min_drop_percent:
                meets_criteria = False
            
            if min_abs_drop is not None and peak['abs_drop'] < min_abs_drop:
                meets_criteria = False
            
            if meets_criteria:
                potential_peaks.append(peak)
        
        print(f"Found {len(all_minima)} local minima, {len(potential_peaks)} meet threshold criteria")
        
        if potential_peaks:
            max_abs_drop = max([p['abs_drop'] for p in potential_peaks])
            max_rel_drop = max([p['drop_percent'] for p in potential_peaks])
            
            for peak in potential_peaks:
                norm_abs = peak['abs_drop'] / max_abs_drop
                norm_rel = peak['drop_percent'] / max_rel_drop
                
                peak['hybrid_score'] = (norm_abs + norm_rel) / 2
            
            potential_peaks.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        if proximity_threshold > 0 and potential_peaks:
            peaks_by_index = sorted(potential_peaks, key=lambda x: x['index'])
            filtered_peaks = [peaks_by_index[0]]
            
            for peak in peaks_by_index[1:]:
                last_kept = filtered_peaks[-1]
                if peak['index'] - last_kept['index'] <= proximity_threshold:
                    if peak['hybrid_score'] > last_kept['hybrid_score']:
                        filtered_peaks[-1] = peak
                else:
                    filtered_peaks.append(peak)
            
            filtered_peaks.sort(key=lambda x: x['hybrid_score'], reverse=True)
            potential_peaks = filtered_peaks
            
            print(f"After proximity filtering (threshold={proximity_threshold} points): {len(potential_peaks)} peaks")
        
        if max_peaks > 0:
            display_peaks = potential_peaks[:max_peaks]
        else:
            display_peaks = []
        
        ax.plot(wavelengths, rolling_avg, 'k--', alpha=0.5, linewidth=0.8, 
                label='Local Background')
        
        for peak in display_peaks:
            ax.plot(peak['wavelength'], peak['intensity'] - marker_offset, 'r^', markersize=8)
            
            ax.text(peak['wavelength'], peak['intensity'] - marker_text_offset, 
                   f"{peak['wavelength']:.1f}", 
                   ha='center', fontsize=FONT_SIZES['xlarge'])
        
        absorption_peaks = potential_peaks
        
        criteria_desc = f"≥{min_drop_percent}% drop"
        if min_abs_drop is not None:
            criteria_desc += f" or ≥{min_abs_drop:.4f} absolute drop"
            
        print(f"Found {len(potential_peaks)} absorption peaks with {criteria_desc}:")
        for i, peak in enumerate(potential_peaks[:max_peaks]):
            print(f"{i+1}. Wavelength: {peak['wavelength']:.2f} nm, Drop: {peak['drop_percent']:.2f}%, "
                  f"Abs Drop: {peak['abs_drop']:.4f}, Score: {peak['hybrid_score']:.2f}")
        if len(potential_peaks) > max_peaks:
            print(f"... and {len(potential_peaks) - max_peaks} more")
    
    ax.set_xlabel('Wavelength (nm)', fontsize=FONT_SIZES['xlarge'])
    ax.set_ylabel('Intensity (arb. units)', fontsize=FONT_SIZES['xlarge'])

    ax.set_xlim(wavelengths[0], wavelengths[-1])
    if title:
        ax.set_title(title)
    
    if label or find_absorption:
        ax.legend(fontsize=FONT_SIZES['xlarge'])
    
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    
    fig.tight_layout()

    print(f"saving to {file_path.split('/')[-1]}.pdf")
    fig.savefig(f"./lab5/{file_path.split('/')[-1]}.pdf", dpi=300)
    
    if show_plot:
        plt.show()
    
    return (fig, ax, absorption_peaks) if find_absorption else (fig, ax)

if __name__ == "__main__":
    file_names_indices = {
    1: "blueskywith400umcable.txt",
    2: "blueskywithmarkers.txt",
    3: "directlyupwithmarkers.txt",
    4: "sunspecwith400umcable.txt",
    5: "sunspecwithmarkers.txt",
    6: "WhiteLEDLamp.txt",
    7: "LED1Blue.txt",
    8: "LED2Green.txt",
    9: "LED3Red.txt",
    10: "LED4Red2Color.txt",
    }

    # Example usage with larger plot and no title
    # fig, ax, peaks = plot_spectrum(
    #     f"{spectrum_data_folderpath}/{file_names_indices[6]}", 
    #     wavelength_min=300, 
    #     wavelength_max=800, 
    #     color_idx=0,
    #     show_plot=False,
    #     title="Spectrum of White LED Lamp",  # Remove title
    #     find_absorption=True,
    #     window_size=25,
    #     min_drop_percent=4.0,
    #     min_abs_drop=0.01,
    #     max_peaks=0,
    #     proximity_threshold=5,
    #     marker_offset=0.01,
    #     marker_text_offset=0.05
    # )

    # fig, ax, peaks = plot_spectrum(
    #     f"{spectrum_data_folderpath}/{file_names_indices[1]}", 
    #     wavelength_min=200, 
    #     wavelength_max=1000, 
    #     color_idx=0,
    #     show_plot=False,
    #     title="Spectrum of Blue Sky with 400 $\mu m$ cable", 
    #     find_absorption=True,
    #     window_size=25,
    #     min_drop_percent=4.0,
    #     min_abs_drop=0.01,
    #     max_peaks=12,
    #     proximity_threshold=5,
    #     marker_offset=0.01,
    #     marker_text_offset=0.05
    # )

    # fig, ax, peaks = plot_spectrum(
    #     f"{spectrum_data_folderpath}/{file_names_indices[2]}", 
    #     wavelength_min=200, 
    #     wavelength_max=1000, 
    #     color_idx=0,
    #     show_plot=False,
    #     title="Spectrum of Blue Sky with 50 $\mu m$ cable",  
    #     find_absorption=True,
    #     window_size=25,
    #     min_drop_percent=4.0,
    #     min_abs_drop=0.01,
    #     max_peaks=12,
    #     proximity_threshold=5,
    #     marker_offset=0.01,
    #     marker_text_offset=0.05
    # )

    fig, ax, peaks = plot_spectrum(
        f"{spectrum_data_folderpath}/{file_names_indices[4]}", 
        wavelength_min=200, 
        wavelength_max=1000, 
        color_idx=0,
        show_plot=False,
        title="Spectrum of Sun with 400 $\mu m$ cable",
        find_absorption=True,
        window_size=25,
        min_drop_percent=4.0,
        min_abs_drop=0.01,
        max_peaks=21,
        proximity_threshold=5,
        marker_offset=0.01,
        marker_text_offset=0.05
    )
