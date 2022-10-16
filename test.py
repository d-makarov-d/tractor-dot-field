import unittest
import asyncio

from web import SiteTree


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
