import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from skimage import exposure
import satdepth.src.utils.satdepth_utils as satdepth_utils
from epimask.external.satdepth.src.utils.useful_methods import timeit


NUM_MATCH_LINES = 40 #20 for plotting 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt='f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} current value- {val:' + self.fmt + '} (running mean- {avg:' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def make_dsm_gridpts(dsm_file:str, 
                     dsm_shrink_buffer:int, 
                     patch_size:int):
    """
    Make patch_size windows on the dsm in the central area with a single side buffer of dsm_shrink_buffer
    -----------------------
    |    ____________     |
    |   | x | x | x |     |
    |   -------------     |
    |   | x | x | x |     |
    |   -------------     |
    |   | x | x | x |     |
    |   -------------     |
    |_____________________|

    Args:
        dsm_file: path to dsm file
        dsm_shrink_buffer: buffer to shrink the dsm from all sides
        patch_size: size of the patch

    Return: 
        list of center coordinates (lat, lon , ht) of the grided points
    """
    dsm_cam = satdepth_utils.ReadDSM(dsm_file)
    dsm = dsm_cam.ReadImg()
    nrows, ncols = dsm.shape

    num_cells_x = (ncols - 2 * dsm_shrink_buffer) // patch_size
    num_cells_y = (nrows - 2 * dsm_shrink_buffer) // patch_size

    # get center coordinates and corresponding lat, lon , ht
    grid_pts = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            center_x = (patch_size) * i + (patch_size // 2) + dsm_shrink_buffer
            center_y = (patch_size) * j + (patch_size // 2) + dsm_shrink_buffer
            if dsm_cam.nodata_mask[int(center_y), int(center_x)]:
                # no data point
                continue
            lon, lat = dsm_cam.backproject(center_y, center_x)
            ht = dsm[center_y, center_x]
            grid_pts.append((lat, lon, ht))

    return grid_pts

@timeit
def plot_matches(test_pair, 
                 matches, 
                 filename, 
                 plt_str=None, 
                 epi_errs=None, 
                 epi_thrs=1.0, 
                 plot_kp=False):
    WHITE_SEPARATION=20
    num_matches, _ = matches.shape
    if num_matches == 0:
        print("No matches found for %s ... moving to next "%(filename))
        return
    if epi_errs != None:
        if len(epi_errs) != num_matches:
            print("Length of epi errors (%d) is not the same as number of matches (%d)"%(len(epi_errs), num_matches))
            return
        mask = np.array(epi_errs) < epi_thrs
    else:
        raise ValueError("epi_errs is None")
    img0_ds, img1_ds, intersection_angle = test_pair
    filename = "%s.pdf" % (filename)

    img0 = img0_ds.ReadImg(repeat=False)
    img1 = img1_ds.ReadImg(repeat=False)

    img0 = exposure.equalize_hist(img0)
    img1 = exposure.equalize_hist(img1)

    h0,w0 = img0.shape
    h1, w1 = img1.shape
    img = np.ones((max(h0,h1), w0+w1+WHITE_SEPARATION))
    img[:h0,:w0] = img0
    img[:h1, w0+WHITE_SEPARATION:] = img1

    num_tp = mask.sum()
    num_fp = num_matches - num_tp
    precision = (num_tp/num_matches) * 100
    textstr = "P: %0.2f  N: %d"%(precision, num_matches)

    cmap_tp = matplotlib.cm.get_cmap("cool")
    cmap_fp = matplotlib.cm.get_cmap("Wistia")
    colors_tp = cmap_tp(np.arange(num_tp)/num_tp)
    colors_fp = cmap_fp(np.arange(num_fp)/num_fp)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    s = 5
    marker_lw = 0.5
    lw = 0.5
    alpha=0.7
    # plot the points
    if plot_kp:
        plt.scatter(matches[mask,0], matches[mask,1], color=colors_tp,s=s, marker='o', alpha=alpha)
        plt.scatter(matches[~mask,0], matches[~mask,1], color=colors_fp,s=s, marker='x', alpha=alpha, linewidths=marker_lw)
        plt.scatter(matches[mask, 2]+ w0 + WHITE_SEPARATION, matches[mask, 3] , color=colors_tp, s=s, marker='o', alpha=alpha)
        plt.scatter(matches[~mask, 2]+ w0 + WHITE_SEPARATION, matches[~mask, 3] , color=colors_fp, s=s, marker='x', alpha=alpha, linewidths=marker_lw)
    
    # plot random true matches using line
    # idx_lines = np.random.permutation(np.arange(num_matches))[:NUM_MATCH_LINES]
    idx_true_matches = np.where(mask)[0]
    idx_lines = np.random.permutation(idx_true_matches)[:NUM_MATCH_LINES]
    for i in idx_lines:
        plt.plot( (matches[i,0], matches[i,2]+w0+WHITE_SEPARATION), (matches[i,1], matches[i,3]), color="#08FF08" , linewidth=lw)

    # put text box and put precision, Num matches
    # text box on top left corner
    props = dict(boxstyle='round', facecolor='red', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color='yellow',
            verticalalignment='top', bbox=props)
    # put plt_str on bottom
    if plt_str:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
        plt.text(0.05, 0.05, plt_str, transform=plt.gca().transAxes,
                 fontsize=10, fontweight='bold', color='yellow',
                 verticalalignment='bottom', bbox=props)

    plt.savefig(filename,dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close("all")
    return
