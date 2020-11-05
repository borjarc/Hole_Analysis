import math as mt
from copy import deepcopy

import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.widgets import TextBox, Button, CheckButtons
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors


def place_window():
    mngr = plt.get_current_fig_manager()
    # get the QTCore PyRect object
    geom = mngr.window.geometry()
    x, y, dx, dy = geom.getRect()
    mngr.window.setGeometry(500, y, dx, dy)


def clear_ax(ax):
    ax.clear()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])


def fit_function(inputs, *params):
    """ Function used to fit the circle. We use the 'squared' version to avoid negative numbers"""
    x, y = inputs
    r = params[0]
    xc = params[1]
    yc = params[2]
    return r - ((x - xc) ** 2 + (y - yc) ** 2)**0.5


def fit_circles_on_holes(contours):
    """ This function fits every contour with a circle and updates each contour with the circle center (xc, yc) and
        radius (R). In addition it computes the residuals, i.e R-R(x,y) for each point (x,y) of the contour."""
    for i, contour in enumerate(contours):
        area = contour.get('area')
        x = np.array(contour.get('x'))
        y = np.array(contour.get('y'))
        initial_guess = [(area / np.pi) ** 0.5, np.mean(x), np.mean(y)]
        ps, p_cov = curve_fit(fit_function, (x, y), np.zeros(x.shape), p0=initial_guess)
        res = fit_function((x, y), *ps)
        contour.update({'xc': ps[1], 'yc': ps[2], 'r': ps[0], 'residuals': res.tolist()})


def sub_image_analysis(img, r_estimated, val=0.5):
    """
    :param img: Cropped image around a single hole
    :param r_estimated: the radius of the fitted circles on the contours found by image processing
    :param val: Define where we look for the edge. Number between 0 and 1
    :return: a list of data points (x,y) that are considered as the edge of the hole
    """

    r_in = mt.floor(0.75*r_estimated)  # The edge of the hole should be between 75% of the estimated radius and the
    xc_int = yc_int = r_out = int((img.shape[0]) * 0.5)  # edge of the image
    n_angle = int(0.8*2*np.pi*r_estimated)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    angles = np.arange(0, 2 * np.pi, step=np.pi * 2 / n_angle)

    line_list = []
    grey_scale_list = []
    for i, angle in enumerate(angles):
        line = []
        grey_scale = []
        x_last = 0
        y_last = 0
        rho = r_in
        x = int(np.round(xc_int + rho * np.cos(angle)))
        y = int(np.round(yc_int + rho * np.sin(angle)))
        while (0 < x < img.shape[1] - 1) and (0 < y < img.shape[0]-1):
            x = int(np.round(xc_int + rho * np.cos(angle)))
            y = int(np.round(yc_int + rho * np.sin(angle)))
            rho += 1
            if not (x == x_last and y == y_last):
                line.append([x, y])
                grey_scale.append(img[y, x])
                x_last = x
                y_last = y
        line_list.append(np.array(line))
        grey_scale_list.append(grey_scale)
    edge_points = []
    for i, grey_scale in enumerate(grey_scale_list):
        if len(grey_scale) > 0:
            derivative = np.diff(np.array(grey_scale, dtype=np.int16), 1)
            diff_percent = 100*derivative/np.array(grey_scale[:-1])
            idx_min = 0
            for j, d in enumerate(diff_percent):
                if d >= 25 and grey_scale[j+1] > grey_scale[j]:
                    idx_min = j + 1
                    break
            idx_max = np.argmax(grey_scale)
            idx = idx_min + mt.ceil(val * (idx_max - idx_min))
            edge_points.append(line_list[i][idx])
    edge_points = np.array(edge_points)
    # len_edge_points = len(edge_points)
    # repeated = len_edge_points - len(np.unique(edge_points, axis=0))
    # print(' Number of edge points found: {}. Number of repeated points: {}'.format(len_edge_points, repeated))
    return edge_points - np.array([xc_int, yc_int])


def get_valid_lattice_constants(distances, indices, tol=20):

    distances_set = list()
    flat_distances = []
    for i in range(len(distances)):
        nn_distances = distances[i]
        for j in range(1, len(nn_distances)):
            if (min([indices[i][0], indices[i][j]]), max([indices[i][0], indices[i][j]])) in distances_set:
                continue
            else:
                distances_set.append((min([indices[i][0], indices[i][j]]), max([indices[i][0], indices[i][j]])))
                flat_distances.append(nn_distances[j])
    med = np.median(flat_distances)
    return [{'d': flat_distances[i], 'index1': distances_set[i][0], 'index2':distances_set[i][1]} for i in
            range(len(flat_distances)) if med - tol <= flat_distances[i] <= med + tol]


def find_nn_distances(coordinates):
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coordinates)
    # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coordinates)
    return nbrs.kneighbors(coordinates)


class SEMAnalyzer:
    def set_image_name(self):
        self.image_name = self.image_path.split('/')[-1]

    def __init__(self, image, image_path,lattice_const=0.42, radii=[0.1], figsize=(15, 12)):
        self.image_path = image_path
        self.image_name = ''
        self.set_image_name()
        self.raw_image = image.copy()
        self.image = None
        self.lattice_const = lattice_const
        self.radii = radii
        self.img_processed_contours = []
        self.edge_contours = []
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Analyzing: ' + self.image_name)
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.set_xticks([])
        self.ax3.set_xticklabels([])
        self.ax3.set_yticklabels([])
        self.ax3.set_yticks([])
        self.aximage1 = None
        self.aximage2 = None
        self.scatter = None
        self.cbar = None
        self.circle_artists = []
        self.centers_lines = []
        self.lattice_constants = []
        self.conversion_factor = None
        place_window()

    def process_image(self, val):
        """
        :param val: a threshold value
        :return: the processed image which is the result of thresholding, opening and dilatation operations
        """
        tresh = cv2.bitwise_not(cv2.threshold(self.raw_image, val, 255, cv2.THRESH_BINARY)[1])
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        mask_open = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel, iterations=3)
        mask_dilated = cv2.dilate(mask_open, kernel, iterations=1)
        return mask_dilated

    def find_contours(self, img, q):
        """ :param img: processed thresholded image
            :param q: quantile for the area distribution. If the area within the contour is smaller than the q-th
            quantile, the contour is filtered out

            :return: edge image of the holes found by the algorithms. In addition the contours of the SEMAnalyzer are
            updated in this function."""

        img = cv2.GaussianBlur(img, (5, 5), 2)
        mask_edges = cv2.Canny(img, 200, 300, L2gradient=True)
        _, contours, hierarchy = cv2.findContours(mask_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for idx, c in enumerate(contours) if hierarchy[0][idx][3] < 0]
        self.img_processed_contours = []
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            min_area = np.quantile(areas, q)

            for idx, contour in enumerate(contours):
                if areas[idx] > min_area:
                    x, y = zip(*[(c[0][0], c[0][1]) for c in contour])
                    if (0 in x) or (self.raw_image.shape[1]-1 in x) or (0 in y) or (self.raw_image.shape[0]-1 in y):
                        continue
                    self.img_processed_contours.append({'x': list(x), 'y': list(y), 'area': areas[idx]})

    def find_threshold_value(self):
        """Returns the threshold value as the 20%-ile of the image"""
        img_ravel = self.raw_image.ravel()
        return np.percentile(img_ravel, 20)

    def draw_contours(self):
        """ This function just draw the different contours on the image and the resulting fitted results. It outputs
            the image with the contours, circles and centers drawn on it. To see it we need to 'imshow' the returned
            image.
        """
        img = self.raw_image.copy()
        for art in self.circle_artists:
            art.remove()
        self.circle_artists = []
        for l in self.centers_lines:
            l.remove()
        self.centers_lines = []
        for contour in self.img_processed_contours:
            c = np.array([[contour['x'][i], contour['y'][i]] for i in range(len(contour['x']))])
            cv2.drawContours(img, [c], -1, (255, 0, 0), 1)
            xc = contour.get('xc')
            yc = contour.get('yc')
            radius = contour.get('r')
            circle = Circle((xc, yc), radius, color='blue', fill=None)
            self.circle_artists.append(self.ax1.add_artist(circle))
            l, = self.ax1.plot(xc, yc, 'o', color=[0, 0, 1], markersize=3)
            self.centers_lines.append(l)
        # ===============================
        for contour in self.edge_contours:
            ll, = self.ax1.plot(contour['x'], contour['y'], 'o', color=[1, 1, 0], markersize=3, alpha=0.9)
            self.centers_lines.append(ll)
            xc = contour.get('xc')
            yc = contour.get('yc')
            radius = contour.get('r')
            circle = Circle((xc, yc), radius, color='yellow', fill=None)
            self.circle_artists.append(self.ax1.add_artist(circle))
            l, = self.ax1.plot(xc, yc, 'o', color=[1, 1, 0], markersize=3, fillstyle='none')
            self.centers_lines.append(l)
        for lattice in self.lattice_constants:
            l, = self.ax1.plot([lattice['p1'][0], lattice['p2'][0]], [lattice['p1'][1], lattice['p2'][1]], 'r',
                               alpha=0.3)
            self.centers_lines.append(l)
        return img

    def detect_holes_edge(self, img, val):
        self.edge_contours = []
        print('start detect_holes_edge...')
        ##################################################
        len_cdl = len(self.img_processed_contours)
        for i, c in enumerate(self.img_processed_contours):
            print('\ranalysing sub-image {}/{} [{:.2f}%]. '.format(i + 1, len_cdl, (i + 1) / len_cdl * 100), end='')
            xc, yc, r = c['xc'], c['yc'], c['r']
            r_out_int = mt.ceil(1.5 * r)
            xc_int, yc_int = int(xc), int(yc)

            # extract a sub image containing one hole
            while (yc_int - r_out_int < 0) or (yc_int + r_out_int > self.raw_image.shape[0]) or \
                    (xc_int - r_out_int < 0) or (xc_int + r_out_int > self.raw_image.shape[1]):
                r_out_int -= 1
            img_sub = img[(yc_int - r_out_int):(yc_int + r_out_int + 1), (xc_int - r_out_int):(xc_int + r_out_int + 1)]

            ##################################################
            if int(2*np.pi*r) > 10:
                edges = sub_image_analysis(img_sub, r, val=val) + np.array([xc_int, yc_int])
                x, y = zip(*edges)
                self.edge_contours.append({'x': list(x), 'y': list(y), 'area': cv2.contourArea(edges)})
        fit_circles_on_holes(self.edge_contours)

    def find_nearest_neighbors(self):
        if len(self.edge_contours) > 7:
            coordinates = np.array([[c['xc'], c['yc']] for c in self.edge_contours])
            distances, indices = find_nn_distances(coordinates)
            self.lattice_constants = get_valid_lattice_constants(distances, indices)
            self.set_conversion_factor()

            for x in self.lattice_constants:
                p1 = coordinates[x['index1']]
                p2 = coordinates[x['index2']]
                center = 0.5*(p1 + p2)

                x.update({'p1': p1, 'p2': p2, 'center': center})
        else:
            self.lattice_constants = []

    def set_conversion_factor(self):
        self.conversion_factor = self.lattice_const/np.median([x['d'] for x in self.lattice_constants])

    def draw_distributions(self):
        px = []
        py = []
        ds = []
        for x in self.lattice_constants:
            p1 = x['p1']
            p2 = x['p2']
            center = 0.5 * (p1 + p2)
            px.append(center[0])
            py.append(self.raw_image.shape[1] - center[1])
            ds.append(x['d'] * self.conversion_factor)
        if self.cbar:
            self.cbar.remove()
        self.scatter = self.ax3.scatter(px, py, c=ds, cmap='jet')
        self.cbar = self.fig.colorbar(self.scatter, ax=self.ax3)
        self.cbar.set_label(r'Lattice constant distance [$\mu$m]')
        delta_a = np.round(100*(max(ds) - min(ds))/self.lattice_const, 2)
        self.ax3.set_title(r'$\Delta$a/a$_{design}$ % = ' + str(delta_a))
        self.ax4.clear()
        radii = np.array([x['r'] for x in self.edge_contours])*self.conversion_factor
        for r in self.radii:
            self.ax4.axvline(r, color='red')
        dr = (max(radii) - min(radii))/len(radii)
        self.ax4.hist(radii, bins=np.arange(min(radii)-2*dr, max(radii)+2*dr, dr))
        self.ax4.set_title('mean {:.4f}, std {:.4f}'.format(np.mean(radii), np.std(radii)))
        self.ax4.set_xlabel(r'Radii distribution [$\mu$m]')

    def recompute(self, threshold_val, quant, edge_thresh, perform_edge_detection):
        dilated_treshold = self.process_image(threshold_val)
        self.find_contours(dilated_treshold, quant)
        fit_circles_on_holes(self.img_processed_contours)
        if perform_edge_detection:
            # Depending on the matplotlib backend image encoding is RGB or BGR
            self.detect_holes_edge(cv2.cvtColor(self.raw_image.copy(), cv2.COLOR_RGB2GRAY), edge_thresh)
            self.find_nearest_neighbors()
            self.draw_distributions()
        else:
            self.edge_contours = []
            self.lattice_constants = []
            if self.cbar:
                self.cbar.remove()
                self.cbar = None
                clear_ax(self.ax3)
            self.ax4.clear()
        img_with_contours = self.draw_contours()
        self.aximage2.set_data(dilated_treshold)
        self.aximage1.set_data(img_with_contours)

        self.fig.canvas.draw_idle()

    def generate_window(self):
        """ This function should be called to initiate the analysis of a given image. Each time the user changes a
            a value in the interface the 'recompute()' method should be called"""
        dilated_treshold = self.process_image(self.find_threshold_value())
        self.find_contours(dilated_treshold, 0)
        fit_circles_on_holes(self.img_processed_contours)
        img_with_contours = self.draw_contours()
        self.aximage2 = self.ax2.imshow(dilated_treshold)
        self.aximage1 = self.ax1.imshow(img_with_contours)

        # ========= Defines the text button for the threshold value =================
        def submit(text):
            if not text:
                threshold_value = 0
            else:
                threshold_value = eval(text)
            quant = eval(text_box_area.text)
            edge_threshold = eval(text_box_edge.text)
            perform_edge_detection = check_edge.get_status()[0]
            self.radii = [eval(x) for x in text_box_radii.text.split(',') if x != '']
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)

        axbox_thresh = self.fig.add_axes([0.5, 0.93, 0.03, 0.023])
        text_box_thresh = TextBox(axbox_thresh, 'Threshold value (0-255): ', initial=str(self.find_threshold_value()))
        text_box_thresh.on_submit(submit)
        # ===========================================================================

        # ========= Defines the text button for the area filtering ==================
        def submit_area_threshold(text):
            if not text:
                quant = 0
            else:
                quant = eval(text)
            threshold_value = eval(text_box_thresh.text)
            edge_threshold = eval(text_box_edge.text)
            perform_edge_detection = check_edge.get_status()[0]
            self.radii = [eval(x) for x in text_box_radii.text.split(',') if x != '']
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)

        axbox_area = self.fig.add_axes([0.5, 0.88, 0.03, 0.023])
        text_box_area = TextBox(axbox_area, 'Area selection (0-1): ', initial=str(0))
        text_box_area.on_submit(submit_area_threshold)
        # ===========================================================================

        # ======= Defines the text button for holes edge detection =================
        def submit_edge_detect(text):
            if not text:
                edge_threshold = 0
            else:
                edge_threshold = eval(text)
            threshold_value = eval(text_box_thresh.text)
            quant = eval(text_box_area.text)
            perform_edge_detection = check_edge.get_status()[0]
            self.radii = [eval(x) for x in text_box_radii.text.split(',') if x != '']
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)

        axbox_edge = self.fig.add_axes([0.8, 0.95, 0.03, 0.023])
        text_box_edge = TextBox(axbox_edge, 'Edge threshold (0-1): ', initial=str(1))
        text_box_edge.on_submit(submit_edge_detect)

        # ===========================================================================

        # ======= Defines the check button for holes edge detection =================

        def check_edge_action(event):
            perform_edge_detection = check_edge.get_status()[0]
            threshold_value = eval(text_box_thresh.text)
            quant = eval(text_box_area.text)
            edge_threshold = eval(text_box_edge.text)
            self.radii = [eval(x) for x in text_box_radii.text.split(',') if x != '']
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)
        checkax_edge = self.fig.add_axes([0.75, 0.9, 0.1, 0.04])
        check_edge = CheckButtons(checkax_edge, ['Edge detection'], [False])
        check_edge.on_clicked(check_edge_action)

        # ===========================================================================

        # ======= Defines the Text button for the lattice constant =================

        def submit_lattice_value(text):
            if not text:
                lattice_const = 0
            else:
                lattice_const = eval(text)
            self.lattice_const = lattice_const
            perform_edge_detection = check_edge.get_status()[0]
            threshold_value = eval(text_box_thresh.text)
            quant = eval(text_box_area.text)
            edge_threshold = eval(text_box_edge.text)
            self.radii = [eval(x) for x in text_box_radii.text.split(',') if x != '']
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)

        axbox_lattice = self.fig.add_axes([0.15, 0.95, 0.03, 0.023])
        text_box_lattice = TextBox(axbox_lattice, 'Lattice constant (um): ', initial=str(0.42))
        text_box_lattice.on_submit(submit_lattice_value)

        # ===========================================================================

        # ======= Defines the Text button for the radii =================

        def submit_radii_value(text):
            self.radii = [eval(x) for x in text.split(',') if x != '']
            perform_edge_detection = check_edge.get_status()[0]
            threshold_value = eval(text_box_thresh.text)
            quant = eval(text_box_area.text)
            edge_threshold = eval(text_box_edge.text)
            self.recompute(threshold_value, quant, edge_threshold, perform_edge_detection)

        axbox_radii = self.fig.add_axes([0.15, 0.9, 0.055, 0.023])
        initial_radii_string = ''
        for rad in self.radii:
            initial_radii_string += str(rad) + ','
        text_box_radii = TextBox(axbox_radii, 'Radii (um): ', initial=initial_radii_string)
        text_box_radii.on_submit(submit_radii_value)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85, wspace=0.1, hspace=0.1)
        plt.show()

    def save_results(self):
        pass

    def get_analysis_results(self):
        if len(self.edge_contours) == 0:
            return None
        phc_name = self.image_name.split('_')[0]
        all_radii_residuals = []
        for x in self.edge_contours:
            res = np.array(x['residuals'])*self.conversion_factor
            all_radii_residuals += res.tolist()
        fitted_lattice_constant = [x['d'] * self.conversion_factor for x in self.lattice_constants]
        return dict(phc_name=phc_name,
                    fitted_radii=[x['r']*self.conversion_factor for x in self.edge_contours],
                    designed_radii=self.radii,
                    median_residuals=np.median(all_radii_residuals),
                    quantile75=np.quantile(all_radii_residuals, 0.75),
                    quantile25=np.quantile(all_radii_residuals, 0.25),
                    radii_std=np.std(all_radii_residuals),
                    lattice_constant=[self.lattice_const] if isinstance(self.lattice_const, float) else self.lattice_const,
                    fitted_lattice_constant=fitted_lattice_constant,
                    lattice_const_std=np.std(fitted_lattice_constant)
                    )


