import argparse
import pathlib
from os import listdir

from db.brick_item import BrickDB, BrickItem
from app_preferences import AppPreferences


def _data_dir_stat(path: str):
    path = pathlib.Path(path)
    print("Data directory: %s" % path.absolute())
    if not path.is_dir():
        print("\tNot yet created")
    else:
        content = listdir(path)
        jpg_files = list(filter(lambda itm: itm.endswith(".jpg"), content))
        fits_files = list(filter(lambda itm: itm.endswith(".fits.gz"), content))
        print("\tcached pictures: %d" % len(jpg_files))
        print("\tfits files: %d" % len(fits_files))


def _db_stat(path_str: str):
    path = pathlib.Path(path_str)
    print("Database: %s" % path.absolute())
    if not path.is_file():
        print("\tNot yet created")
    else:
        db = BrickDB(path_str)
        for status in BrickItem.STATUSES:
            res = db.con.cursor().execute("SELECT COUNT(*) FROM peaks WHERE status == '%s'" % status)
            print("\t%s: %s" % (status, res.fetchone()[0]))


def _print_status():
    # check data dir
    data_dir = 'data'
    db_dile = "main_db.sqlite"
    _data_dir_stat(data_dir)
    _db_stat(str(pathlib.Path(data_dir).parent.joinpath(db_dile).absolute()))


def run(args: list[str], name: str, prefs: AppPreferences):
    parser = argparse.ArgumentParser(description="Print utility status")
    parser.usage = parser.format_usage().replace('usage: %s' % args[0], '%s %s' % (args[0], name))
    parser.parse_args(args[2:])
    _print_status()

