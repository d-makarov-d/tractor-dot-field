from __future__ import annotations

import argparse
from os import listdir
from os.path import join
import sqlite3
from astropy.io import fits
from astropy.io.fits.hdu.table import FITS_rec
import re
import numpy as np
import numbers

from app_preferences import AppPreferences
from db.brick_item import BrickDB


table_name = "data"


def run(args: list[str], name: str, prefs: AppPreferences):
    parser = argparse.ArgumentParser(description="Extract data from fits files to SQLite database")
    parser.usage = parser.format_usage().replace('usage: %s' % args[0], '%s %s' % (args[0], name))

    parser.add_argument("--out", "-o", type=str,
                        help="Output SQLite file", required=True)
    parser.add_argument("--columns", type=str, nargs="*",
                        help="Columns to extract", required=True)

    settings = parser.parse_args(args[2:])

    _extract_data(settings.out, settings.columns, prefs.data_dir)


def _extract_type(name: str, data) -> TypeDescr:
    def gues_dtype(v0):
        if isinstance(v0, numbers.Integral):
            return "INTEGER"
        elif isinstance(v0, numbers.Real):
            return "REAL"
        else:
            return "TEXT"

    if isinstance(data, np.ndarray):
        return TypeDescr(name, gues_dtype(data[0]), True, data)
    else:
        return TypeDescr(name, gues_dtype(data), False, data)


def _create_table(columns: list[TypeDescr], db):
    cmd = f"CREATE TABLE IF NOT EXISTS {table_name}("
    for column in columns:
        cmd = cmd + f"{column.name} {column.var_type}, "
    cmd = cmd[:-2] + ")"
    db.execute(cmd)


def _construct_query(columns: list[TypeDescr], table: str) -> str:
    cmd = f"INSERT INTO {table}("
    for column in columns:
        cmd = cmd + f"{column.name}, "
    cmd = cmd[:-2]
    cmd = cmd + ") VALUES(" + len(columns) * "?, "
    cmd = cmd[:-2] + ")"
    return cmd


def _put_fits(columns: list[str], file: str, db):
    try:
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = hdul[1].data
            if not isinstance(data, FITS_rec):
                raise ValueError(f"Fits file {file} must be a table")

            brickid = header['BRICKID']
            brickname = re.sub(r".+/(.*)\.fits.+", r"\g<1>", file)
            types = [_extract_type("brickid", brickid), _extract_type("brickname", brickname)]
            for column in columns:
                types.append(_extract_type(column, data[column]))
            _create_table(types, db)
            query = _construct_query(types, table_name)
            arr = list(filter(lambda it: it.is_arr, types))[0]
            cur = db.cursor()
            for i in range(0, len(arr.data)):
                cur.execute(query, [it.get(i) for it in types])
            db.commit()
    except Exception as e:
        print(f"error in {file} [{e}]")


def _extract_data(out: str, columns: list[str], data_dir: str):
    fitses = [join(data_dir, f) for f in listdir(data_dir) if (join(data_dir, f)).endswith("fits.gz")]
    print(f"processing {len(fitses)}")
    db = sqlite3.connect(out)
    for i, fits in enumerate(fitses):
        _put_fits(columns, fits, db)
        print("\r%.2f%%" % (i / float(len(fitses)) * 100), end="")


class TypeDescr:
    def __init__(self, name: str, var_type: str, is_arr: bool, data):
        self.var_type = var_type
        self.is_arr = is_arr
        self.name = name
        self.data = data

    def get(self, i: int):
        if self.is_arr:
            return self._adapt_np_types(self.data[i])
        else:
            return self._adapt_np_types(self.data)

    def _adapt_np_types(self, v):
        if isinstance(v, np.ndarray):
            return str(list(v)).replace("  ", ",")
        elif self.var_type == "REAL":
            return float(v)
        elif self.var_type == "INTEGER":
            return int(v)
        return v
