import sqlite3

from _db_abc import DBInstance, DBModel
from main import BrickResult

"""Class, operating bricks database"""
class BrickDB:
    def __init__(self, file: str):
        """
        :param file: Database file
        """
        self._db = sqlite3.connect(file)
        self._results: Collection = self._db[self.collection_name]

    def insert(self, brick_name: str, result: Collection[BrickResult]) -> bool:
        """
        Inserts analysis result into the database
        :param brick_name: Name of analyzed brick
        :param result: Peak analysis result
        :return: True if the operation scucceeded
        """
        self._results

class BrickItem(DBInstance):
    def __int__(self, id: str, ):
        TableDesr.Field('brick_name', float),
        TableDesr.Field('x', float),
        TableDesr.Field('y', float),
        TableDesr.Field('n_points', float),
        TableDesr.Field('area', float),
        TableDesr.Field('ra', float),
        TableDesr.Field('dec', float),
        TableDesr.Field('x', bytes),
        TableDesr.Field('y', bytes),
        TableDesr.Field('Z', bytes),
        TableDesr.Field('mask_inside', bytes),
        TableDesr.Field('mask_area', bytes),


    @staticmethod
    def from_db(data: tuple) -> BrickItem:
        return BrickItem()

    def to_tuple(self) -> tuple:
        return (
            self._id,
            self.dist,
            self.ra,
            self.dec,
            self.mass,
            self.ed
        )

    @staticmethod
    def table_descr() -> TableDesr:
        fields = [
            TableDesr.Field('brick_name', str),
            TableDesr.Field('x', float),
            TableDesr.Field('y', float),
            TableDesr.Field('n_points', float),
            TableDesr.Field('area', float),
            TableDesr.Field('ra', float),
            TableDesr.Field('dec', float),
            TableDesr.Field('x', bytes),
            TableDesr.Field('y', bytes),
            TableDesr.Field('Z', bytes),
            TableDesr.Field('mask_inside', bytes),
            TableDesr.Field('mask_area', bytes),
        ]
        _id = TableDesr.Field('id', str)
        return TableDesr('peaks', fields, _id)
