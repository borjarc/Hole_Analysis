import argparse
import textwrap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import os
from imaging.SEM.analysis_script import perform_batch_analysis


def get_directory_path():

    app = QApplication(sys.argv)
    dir_dialog = QFileDialog()
    default_dir = '/Users/nicolasvilla/Thesis/fabrication/process_runs'
    d = dir_dialog.getExistingDirectory(dir_dialog,
                                        caption='Select directory',
                                        directory=default_dir,
                                        options=dir_dialog.DontResolveSymlinks)
    QTimer.singleShot(100, app.quit)
    app.exec_()
    if d == '':
        sys.exit()
    else:
        return d


def isimagefile(file):
    if file.endswith('.tif') or file.endswith('.jpg'):
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("Script for analysing SEM images"))

    parser.add_argument('--lattice_constant', help='Nomnal value of the lattice constant in um', type=float, default=0.42)
    parser.add_argument('--radii', help='Nominal values of the holes radi in um', nargs='+', type=float, default=[0.1])

    args = parser.parse_args()

    all_files_are_images = False
    dir_path = None
    while not all_files_are_images:
        dir_path = get_directory_path()
        all_files_are_images = all([isimagefile(file) for file in os.listdir(dir_path) if file.endswith('.jpg') or file.endswith('.tif')])
        if not all_files_are_images:
            print("Not all files are image files. Please select a correct folder or cancel the selection")
    image_file_names = sorted([dir_path + '/' + file for file in os.listdir(dir_path) if file.endswith('.jpg') or file.endswith('.tif')])
    perform_batch_analysis(image_file_names, args.lattice_constant, args.radii)


if __name__ == '__main__':

    main()
