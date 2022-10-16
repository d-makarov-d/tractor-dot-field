import numpy as np
from astropy.io.fits.hdu.table import FITS_rec
from astropy.io.fits.hdu.base import Header

"""Class, holding data, read from brick"""
class Brick:
    def __init__(self, header: Header, data: FITS_rec):
        self._brickid: int = header['BRICKID']
        self._release: np.ndarray = data['RELEASE']
        self._brickname: np.chararray = data['BRICKNAME']
        self._objid: np.ndarray = data['OBJID']
        self._type: np.chararray = data['TYPE']
        self._ra: np.ndarray = data['RA']
        self._dec: np.ndarray = data['DEC']
        self._bx: np.ndarray = data['BX']
        self._by: np.ndarray = data['BY']
        self._flux_g: np.ndarray = data['FLUX_G']
        self._flux_r: np.ndarray = data['FLUX_R']
        self._flux_z: np.ndarray = data['FLUX_Z']
        self._flux_ivar_g: np.ndarray = data['FLUX_IVAR_G']
        self._flux_ivar_r: np.ndarray = data['FLUX_IVAR_R']
        self._flux_ivar_z: np.ndarray = data['FLUX_IVAR_Z']

    def get_points(self, type: str = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates of points inside the brick
        :param type: Type of points, all are returned if type is None
        :return: Tuple(bx, by)
            bx: X - coordinates
            by: Y - coordinates
        """
        if type is None:
            return (self._bx, self._by)

        mask = self._type == type
        return (self._bx[mask], self._by[mask])

    def get_skycoords(self, type: str = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates of points on the sky
        :param type: Type of points, all are returned if type is None
        :return: Tuple(ra, dec)
            ra: X - ...
            dec: Y - ...
        """
        if type is None:
            return (self._ra, self._dec)

        mask = self._type == type
        return (self._ra[mask], self._dec[mask])