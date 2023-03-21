import argparse
import re
from typing import Union, Callable, Coroutine, Collection
import asyncio
from functools import reduce
from os import listdir
import pathlib
import time

from web import SiteTree
from app_preferences import AppPreferences

urls = [
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/006/tractor-0064m110.fits',
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/012/tractor-0126p330.fits',
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/149/tractor-1491p690.fits',
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/155/tractor-1554p180.fits',
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/189/tractor-1896p327.fits',
    'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/tractor/211/tractor-2118p350.fits',
]


def run(args: list[str], name: str, prefs: AppPreferences):
    parser = argparse.ArgumentParser(description="Download fits files")
    parser.usage = parser.format_usage().replace('usage: %s' % args[0], '%s %s' % (args[0], name))

    parser.add_argument("--download", "-d", action='store_true', help="Download fits mode")
    parser.add_argument("--regex", type=str,
                        default="https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/(north|south)/tractor/\d+/tractor-.+\.fits",
                        help="Pattern to match downloaded fitses")
    parser.add_argument("--root", type=str,
                        default="https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9",
                        help="Scrapper entry point")
    parser.add_argument("--proxy", type=str,
                        default=None, required=False,
                        help="Proxy")

    settings = parser.parse_args(args[2:])

    tree = SiteTree(proxy=settings.proxy)

    if settings.download:
        print("Searching fitses")
        urls = asyncio.run(_get_urls_by_regex(tree, settings.regex, settings.root))
        print(f"{len(urls)} fitses found")
        actual_urls = _filter_urls(urls, prefs)
        print(f"download tasks: {len(actual_urls)}")
        asyncio.run(_process_urls(actual_urls, prefs, tree))


def _download_urls(
        urls: Collection[str], data_dir: str, tree: SiteTree,
        progress: Callable[[float], None] = None
) -> list[Coroutine]:
    progresses = dict((url, 0.0) for url in urls if _extract_name(url) is not None)
    t = time.time()

    def update_progress(url: str, new_progress: float):
        nonlocal t
        progresses[url] = new_progress
        if time.time() - t > 0.1:
            t = time.time()
            progress(sum(progresses.values()) / len(progresses))

    def make_task(url: str) -> Coroutine:
        name = _extract_name(url)
        file_name = f"{data_dir}/{name}"
        return tree.download(url, file_name, lambda pr: update_progress(url, pr))

    # Using progress keys to be sure _extract_name() result is not None
    tasks = [make_task(url) for url in progresses.keys()]

    return tasks


def _extract_name(url: str) -> Union[str, None]:
    res = re.sub(r'.*/\w+-', '', url)
    if len(res) == 0 or len(res) == len(url):
        return None

    return res


def _filter_urls(urls: list[str], prefs: AppPreferences):
    """Filter out urls for existing files"""
    if not pathlib.Path(prefs.data_dir).is_dir():
        return urls

    content = listdir(prefs.data_dir)
    existing = list(filter(lambda itm: itm.endswith(".fits"), content))

    return [url for url in urls if _extract_name(url) not in existing]


def _print_progress(pr: float):
    print(f"Download progress: %2.4f%%" % (pr * 100), end='\r')


async def _process_urls(urls: Collection[str], prefs: AppPreferences, tree: SiteTree):
    tasks = _download_urls(urls, prefs.data_dir, tree, _print_progress)
    done, _ = await asyncio.wait(tasks)
    success = 0
    error = 0
    for fut in done:
        exception = fut.exception()
        if exception is None:
            success += 1
        else:
            error += 1
    print(f"Successful tasks: {success}, Failed tasks: {error}")


async def _get_urls_by_regex(tree: SiteTree, regex: str, root: str) -> list[str]:
    delimiters_root = root.count('/')
    re_part = reduce(lambda acc, el: acc + '/' + el, regex.split('/')[:(delimiters_root + 2)])
    content = await tree.ls(root)
    results = []
    is_leaf = False
    for part in content:
        new_path = f"{root}/{part}"
        if new_path.endswith("/"):
            new_path = new_path[:-1]
        if re.match(re_part, new_path):
            if re_part == regex:
                is_leaf = True
                results.append(new_path)
            else:
                results.extend(await _get_urls_by_regex(tree, regex, new_path))

    if is_leaf:
        print(f"{root} :\t {len(results)}")

    return results
