import cv2
from imaging.SEM.SEMAnalyzer import SEMAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import json


def analyze(path_to_file, lattice_constant, radii):
    image = cv2.imread(path_to_file)
    sp = SEMAnalyzer(image, path_to_file, lattice_constant, radii)
    sp.generate_window()
    return sp.get_analysis_results()


def plot_analysis_results(results, save_path, save=True):
    xtick_labels = list(results.keys())
    xticks = list(range(1, len(xtick_labels) + 1))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, size=15, rotation=90)
    ax1.set_ylabel(r'Fitted Radii [$\mu$m]', size=15)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels, size=15, rotation=90)
    ax2.set_ylabel(r'Radii Residuals median/ inter quantiles [$\mu$m]', size=15)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xtick_labels, size=15, rotation=90)
    ax3.set_ylabel(r'Lattice constant [$\mu$m]', size=15)
    radii_residuals_stds = []
    lattice_stds = []
    for i, (phc_name, results_list) in enumerate(results.items()):
        idx = i+1
        for result in results_list:
            lattice_stds.append(result['lattice_const_std'])
            radii_residuals_stds.append(result['radii_std'])
            l, = ax1.plot([idx]*len(result['fitted_radii']), result['fitted_radii'], 'o', alpha=0.2, markersize=7)
            ax1.plot([idx]*len(result['designed_radii']), result['designed_radii'], '*', color=l.get_color(),
                     markersize=20)
            ax2.errorbar([idx], result['median_residuals'], yerr=[[abs(result['quantile25'])], [abs(result['quantile75'])]],
                         color=l.get_color(), alpha=0.4, lolims=True, uplims=True, fmt='o')
            ax3.plot([idx]*len(result['fitted_lattice_constant']), result['fitted_lattice_constant'], 'o', alpha=0.2,
                     color=l.get_color(), markersize=7)
            ax3.plot([idx]*len(result['lattice_constant']), result['lattice_constant'], '*', color=l.get_color(),
                     markersize=20)
    ax2.set_title(r'<$\sigma$> = {:.3f}'.format(np.mean(radii_residuals_stds)), size=15)
    ax3.set_title(r'<$\sigma$> = {:.3f}'.format(np.mean(lattice_stds)), size=15)
    plt.tight_layout()
    if save:
        plt.savefig(save_path + 'batch_results.pdf')
    plt.show()


def perform_batch_analysis(image_names, lattice_constant, radii):
    splitted_path = image_names[0].split('/')
    path_to_save_results = ''
    for s in splitted_path[:-1]:
        path_to_save_results += s + '/'
    results = {}
    for file_name in image_names:
        result = analyze(file_name, lattice_constant, radii)
        with open(file_name.split('.')[0] + '.json', 'w') as f:
            json.dump(result, f)
        if result:
            keys = results.keys()
            if result['phc_name'] in keys:
                results[result['phc_name']].append(result)
            else:
                results.update({result['phc_name']: [result]})

    plot_analysis_results(results, path_to_save_results, save=True)

if __name__ == '__main__':
    path = "/Users/nicolasvilla/Thesis/fabrication/process_runs/la0619_tra00/20190705_la0619_tra00/"
    file = "phc1_1.jpg"
    # path = "/Users/nicolasvilla/Thesis/fabrication/process_runs/soitec1/20191105_afterZepStrip/analyzed/"
    # file = "phc4_mid_3.jpg"
    # file = "phc90_lr_2.jpg"

    analyze(path + file, 0.42, [0.1])
