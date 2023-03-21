import asyncio
import re
from os import listdir, remove
from os.path import isfile, join
import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.table import FITS_rec
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from scipy.stats import gaussian_kde
from scipy.ndimage import label
from typing import Collection
import uuid

from brick import Brick
from peak import Peak
from web import SiteTree
from db.brick_item import BrickDB, BrickItem


threshold = 5


class BrickResult:
    """Result of brick pick analysis"""
    def __init__(self, peak: Peak, x: np.ndarray, y: np.ndarray, Z: np.ndarray, mask_inside: np.ndarray, mask_area: np.ndarray,
                 flux_g: np.ndarray, flux_z: np.ndarray, flux_r: np.ndarray):
        self.peak = peak
        self.x = x
        self.y = y
        self.Z = Z
        self.flux_g = flux_g
        self.flux_z = flux_z
        self.flux_r = flux_r
        self._mask_inside = mask_inside
        self.mask_area = mask_area


def decode_fits(file: str) -> Brick:
    with fits.open(file) as hdul:
        header = hdul[0].header
        data = hdul[1].data
        if not isinstance(data, FITS_rec):
            raise ValueError(f"Fits file {file} must be a table")

        return Brick(header, data)


def find_peaks(brick: Brick, points_x=300, points_y=300, sigma=3.0) -> list[BrickResult]:
    """
    Finds meaningful clusters of 'REX' type points
    :param brick: Brick in where to search
    :return: Peaks
    """
    # REX type objects coordinates inside the brick
    x, y = brick.get_points(type=['REX', 'PSF'])
    ra, dec = brick.get_skycoords(type=['REX', 'PSF'])

    def mag(flux):
        return 22.5 - 2.5 * np.log10(flux)

    def filter_mag(f_g, f_r, f_z):
        return ( (26.5 > mag(f_g) > 18) and (25.5 > mag(f_r) > 18) ) \
            or ( (26.5 > mag(f_g) > 18) and (25.5 > mag(f_z) > 17) ) \
            or ( (25.5 > mag(f_r) > 18) and (25.5 > mag(f_z) > 17) )

    x, y, ra, dec, flux_g, flux_r, flux_z = brick.get_properties(
        lambda br: np.array([t in ('REX', 'PSF') and filter_mag(g, r, z) for t, g, r, z in zip(br._type, br._flux_g, br._flux_r, br._flux_z)]),
        lambda br: [br._bx, br._by, br._ra, br._dec, br._flux_g, br._flux_r, br._flux_z]
    )
    # make grid, on which we will search for peaks
    X, Y = np.meshgrid(np.linspace(min(x),max(x), points_x), np.linspace(min(y),max(y), points_y))
    # compute probability density estimates in the grid nodes
    kernel = gaussian_kde(np.vstack([x, y]))
    positions = np.vstack([Y.ravel(), X.ravel()])
    Z = np.rot90(np.reshape(kernel(positions).T, X.shape))

    n_points = len(x)
    S = (max(x) - min(x)) * (max(y) - min(y))

    # check if peak is meaningful
    is_meaningful = lambda kde: \
        (kde - 1 / S) * np.sqrt(n_points * 4 * np.pi * np.linalg.eig(kernel.covariance)[0].prod() ** 0.5 / kde) > threshold

    mask_meaningful = is_meaningful(Z)
    # allow all types of connections
    structure = np.ones((3,3))
    labeled, n_components = label(mask_meaningful, structure)

    results = list()
    for i in range(1, n_components + 1):
        # measure info about found area
        area = labeled == i
        area_y, area_x = np.indices(area.shape)
        area_x = area_x[area == True]
        area_y = area_y[area == True]
        center_x = min(x) + area_x.sum() / len(area_x) / points_x * (max(x) - min(x))
        center_y = max(y) - min(y) - area_y.sum() / len(area_y) / points_y * (max(y) - min(y))
        area_s = len(area_x) / (points_x * points_y) * S
        # r = (((area_x - center_x) ** 2 + (area_y - center_y) ** 2) ** 0.5).max()
        points_idx_x = (x - min(x)) / (max(x) - min(x) - 1) * points_x
        points_idx_y = (1 - (y - min(y)) / (max(y) - min(y) - 1)) * points_y
        mask_points_inside = np.zeros(n_points, dtype=bool)

        def query_area(_x, _y):
            if _x >= 0 and _x < points_x and _y >= 0 and _y < points_y:
                return area[_y, _x]
            else:
                return False
        for point_idx in range(n_points):
            mask_points_inside[point_idx] = \
                query_area(int(np.floor(points_idx_x[point_idx])), int(np.floor(points_idx_y[point_idx]))) or \
                query_area(int(np.ceil(points_idx_x[point_idx])), int(np.floor(points_idx_y[point_idx]))) or \
                query_area(int(np.floor(points_idx_x[point_idx])), int(np.ceil(points_idx_y[point_idx]))) or \
                query_area(int(np.ceil(points_idx_x[point_idx])), int(np.ceil(points_idx_y[point_idx])))

        ra_avg = np.mean(ra[mask_points_inside])
        dec_avg = np.mean(dec[mask_points_inside])
        peak = Peak(center_x, center_y, np.count_nonzero(mask_points_inside), area_s, ra_avg, dec_avg)
        results.append(BrickResult(peak, x, y, Z, mask_points_inside, mask_meaningful, flux_g[mask_points_inside], flux_z[mask_points_inside], flux_r[mask_points_inside]))

    return results


async def display_on_fig(fig: Figure, tree: SiteTree, brick_name: str, res: Collection[BrickResult]):
    """
    Display info about found peaks
    :param fig: matplotlib Figure
    :param tree: Provides access to image server
    :param brick_name: Name of thr brick
    :param res: Brick analysis result
    """
    if len(res) < 1:
        return

    # download image of the sky
    img_path = re.sub(
        r"^.+/tractor-(\d{3})(\dp\d+).+$",
        r"coadd/\g<1>/\g<1>\g<2>/legacysurvey-\g<1>\g<2>-image.jpg",
        brick_name
    )
    img_file = re.sub(r"^(.+\.).+$", r"\g<1>jpg", brick_name)
    await tree.download(img_path, img_file)
    img = mpimg.imread(img_file)

    x, y = res[0].x, res[0].y
    Z = res[0].Z

    fig.clear()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, alpha=0.7, c='g', marker='2')
    # ax.imshow(np.rot90(Z), extent=[x.min(), x.max(), y.min(), y.max()])
    ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()])
    for r in res:
        masked_img = np.ma.masked_where(r.mask_area == False, 0.7 * np.ones_like(r.mask_area))
        cmap = plt.get_cmap('Greens')
        # ax.imshow(masked_img, extent=[x.min(), x.max(), y.min(), y.max()], alpha=0.3, cmap=cmap, clim=[0, 1])

    remove(img_file)

    title_str = f"Brick: {brick_name}"
    info_str = ""
    for r in res:
        peak = r.peak
        info_str += f"{peak}\n"
        r = (peak.area / np.pi) ** 0.5
        c = plt.Circle((peak.x, peak.y), radius=r, linewidth=1, fill=None, color='r')
        ax.add_patch(c)
    plt.title(f"{title_str}\n {info_str}")


if __name__ == "__main__":
    dir = "data"
    fitses = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    tree = SiteTree()
    db = BrickDB("main_db.sqlite")

    for i, brick_name in enumerate(fitses):
        print(float(i)/len(fitses))
        try:
            brick = decode_fits(brick_name)
        except Exception:
            print("!!!")
            continue
        results = find_peaks(brick)

        if len(results) > 0:
            fname = re.sub("\.fits$", ".png", brick_name)
            fname = re.sub(r"^.+/", "out/", fname)
            for r in results:
                peak = r.peak
                db.save(BrickItem(
                    str(uuid.uuid4()), brick.url, brick_name,
                    peak.x, peak.y, peak.n_points, peak.area, peak.ra, peak.dec,
                    r.x, r.y, r.Z, r._mask_inside, r.mask_area,
                    r.flux_g, r.flux_z, r.flux_r,
                    brick.width, brick.height
                ))

    tree.close()
