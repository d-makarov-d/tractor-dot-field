from __future__ import annotations

import numpy as np

from ._db_abc import DBInstance, TableDesr, DBModel


class BrickDB(DBModel):
    @property
    def schema(self) -> tuple[TableDesr]:
        return BrickItem.table_descr(),


class BrickItem(DBInstance):
    STATUSES = (
        "unprocessed",
        "good",
        "bad"
    )

    def __init__(self, id: str, brick_name: str, peak_x: float, peak_y: float, n_points: int, area: float,
                ra: float, dec: float, x: np.ndarray, y: np.ndarray, Z: np.ndarray, mask_inside: np.ndarray,
                mask_area: np.ndarray, status: str = None):
        self._id = id
        self._brick_name = brick_name
        self._peak_x = peak_x
        self._peak_y = peak_y
        self._n_points = n_points
        self._area = area
        self._ra = ra
        self._dec = dec
        self._x = x
        self._y = y
        self._Z = Z
        self._mask_inside = mask_inside
        self._mask_area = mask_area
        if status is not None and status not in self.STATUSES:
            raise ValueError("Status must be on of %s, got %s" % (self.STATUSES, status))
        self._status = status or self.STATUSES[0]

    @staticmethod
    def from_db(data: tuple) -> BrickItem:
        return BrickItem(*data)

    def to_tuple(self) -> tuple:
        return (
            self._id,
            self._brick_name,
            self._peak_x,
            self._peak_y,
            self._n_points,
            self._area,
            self._ra,
            self._dec,
            self._x,
            self._y,
            self._Z,
            self._mask_inside,
            self._mask_area,
            self._status
        )

    @staticmethod
    def table_descr() -> TableDesr:
        fields = [
            TableDesr.Field('brick_name', str),
            TableDesr.Field('peak_x', float),
            TableDesr.Field('peak_y', float),
            TableDesr.Field('n_points', int),
            TableDesr.Field('area', float),
            TableDesr.Field('ra', float),
            TableDesr.Field('dec', float),
            TableDesr.Field('x', np.ndarray),
            TableDesr.Field('y', np.ndarray),
            TableDesr.Field('Z', np.ndarray),
            TableDesr.Field('mask_inside', np.ndarray),
            TableDesr.Field('mask_area', np.ndarray),
            TableDesr.Field('status', str),
        ]
        _id = TableDesr.Field('id', str)
        return TableDesr('peaks', fields, _id)
