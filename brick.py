from __future__ import annotations

import numpy as np
from astropy.io.fits.hdu.table import FITS_rec
from astropy.io.fits.hdu.base import Header
from typing import Collection, Callable

"""Class, holding data, read from brick"""
class Brick:
    def __init__(self, header: Header, data: FITS_rec):
        self._brickid: int = header['BRICKID']
        self.url = header['url']
        self._release: np.ndarray = data['RELEASE']
        self._brickname: np.chararray = data['BRICKNAME']
        self._objid: np.ndarray = data['OBJID']
        self._type: np.chararray = data['TYPE']
        self._ra: np.ndarray = data['RA']
        self._dec: np.ndarray = data['DEC']
        self._bx: np.ndarray = data['BX']
        self._by: np.ndarray = data['BY']
        self.width = 3600
        self.height = 3600
        self._flux_g: np.ndarray = data['FLUX_G']
        self._flux_r: np.ndarray = data['FLUX_R']
        self._flux_z: np.ndarray = data['FLUX_Z']
        self._flux_ivar_g: np.ndarray = data['FLUX_IVAR_G']
        self._flux_ivar_r: np.ndarray = data['FLUX_IVAR_R']
        self._flux_ivar_z: np.ndarray = data['FLUX_IVAR_Z']

    def get_properties(
            self, filter: Callable[[Brick], np.ndarray],
            extract_props: Callable[[Brick], list[np.ndarray]]
    ) -> list[np.ndarray]:
        mask = filter(self)

        props = extract_props(self)
        return [x[mask] for x in props]

    def get_points(self, type: Collection[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates of points inside the brick
        :param type: Type of points, all are returned if type is None
        :return: Tuple(bx, by)
            bx: X - coordinates
            by: Y - coordinates
        """
        if type is None:
            return (self._bx, self._by)

        mask = np.array(list(map(lambda x: x in type, self._type)))
        return (self._bx[mask], self._by[mask])

    def get_skycoords(self, type: Collection[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coordinates of points on the sky
        :param type: Type of points, all are returned if type is None
        :return: Tuple(ra, dec)
            ra: X - ...
            dec: Y - ...
        """
        if type is None:
            return (self._ra, self._dec)

        mask = np.array(list(map(lambda x: x in type, self._type)))
        return (self._ra[mask], self._dec[mask])

    def get_flux(self, type: Collection[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.array(list(map(lambda x: x in type, self._type)))
        return (self._flux_g[mask], self._flux_z[mask], self._flux_r[mask])