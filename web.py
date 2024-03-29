import asyncio
import re
import urllib3
import os
import tempfile
import shutil
from typing import Callable

from concurrent.futures import ThreadPoolExecutor


class SiteTree:
    def __init__(self, proxy: str = None, root_url: str = None):
        if proxy is None:
            self._http = urllib3.PoolManager()
        else:
            self._http = urllib3.ProxyManager(proxy)

        if root_url is None:
            self.root_url = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/"

        self._pool = ThreadPoolExecutor(100)

    async def ls(self, path: str = None) -> list[str]:
        """
        Returns content of a folder on the server
        :param path: Path
        :return: List of folder contents
        """
        url = path or self.root_url
        task = self._pool.submit(lambda: self._http.request('GET', url))

        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def cb(f):
            res: urllib3.HTTPResponse = f.result()
            if res.status != 200:
                # TODO !!!
                raise IOError(f"Can not open url {url}: {res.status}")

            document = res.data.decode("utf-8")
            paths = re.findall(r"<tr>.+?<td>.+href=\"(.+?)\".+?</td>.+?</tr>", document)
            # first reference is to parent directory
            paths.pop(0)
            loop.call_soon_threadsafe(lambda: fut.set_result(paths))

        task.add_done_callback(cb)
        return await fut

    async def download(self, path: str, out_file: str, progress: Callable[[float], None] = None) -> bool:
        url = path if "://" in path else f"{self.root_url}{path}"
        task = self._pool.submit(lambda: self._http.request('GET', url, preload_content=False, retries=False))

        # ensure output folder exists
        folders = re.sub('/.+?$', '', out_file)
        os.makedirs(folders, exist_ok=True)

        tmp_file = tempfile.NamedTemporaryFile().name

        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def cb(f):
            chunk_size = 4096
            res: urllib3.HTTPResponse = f.result()
            if res.status != 200:
                progress(1.0)
                loop.call_soon_threadsafe(lambda: fut.set_exception(IOError(f"Can not open url {url}: {res.status}")))
                return

            try:
                with open(tmp_file, 'wb') as out:
                    size = float(res.length_remaining)
                    while True:
                        data = res.read(chunk_size)
                        if not data:
                            break
                        if progress is not None:
                            progress(1 - res.length_remaining / size)
                        out.write(data)
                shutil.move(tmp_file, out_file)
            except OSError as e:
                progress(1.0)
                loop.call_soon_threadsafe(lambda: fut.set_exception(IOError(f"Error writing file to disk: {e.strerror}")))
                res.release_conn()
                return
            except Exception as e:
                progress(1.0)
                loop.call_soon_threadsafe(lambda: fut.set_exception(e))
                res.release_conn()
                return

            loop.call_soon_threadsafe(lambda: fut.set_result(True))

        task.add_done_callback(cb)

        return await fut

    def close(self):
        self._http.clear()
        self._pool.shutdown()
