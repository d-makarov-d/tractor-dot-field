import unittest
import asyncio
import numpy as np
from numpy.testing import assert_array_equal

from web import SiteTree

from db.brick_item import BrickItem, BrickDB


class Test(unittest.TestCase):
    def test_web_ls(self):
        tree = SiteTree()

        async def task():
            return await tree.ls()

        res = asyncio.run(task())
        tree.close()

        self.assertEqual(298, len(res))

    def test_download_folder(self):
        tree = SiteTree()

        async def task():
            folder = 'tractor/137/'
            fitses = await tree.ls(folder)
            fitses.pop(0)

            tasks = [tree.download(folder+fits, 'data/'+fits) for fits in fitses]
            tasks = [asyncio.create_task(task) for task in tasks]
            await asyncio.wait(tasks)

        res = asyncio.run(task())
        tree.close()


class DBTest(unittest.TestCase):
    def test_insert_brick_result(self):
        db = BrickDB(":memory:")
        bi = BrickItem("id1", "brick_name", 0.1, 0.1, 1, 0.1, 0.1, 0.1,
                       np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))
        db.save(bi)
        assert_array_equal(db.find(BrickItem)[0]._mask_area, np.array([1, 2, 3]))
