import re
from astropy.io import fits
from astropy.io.fits.hdu.table import FITS_rec

from web import SiteTree
import asyncio

names = []
with open('human-picked.dat') as f:
    for l in f.readlines()[2:]:
        columns = l.split('|')
        columns = list(map(lambda x: x.strip(), columns))
        name = columns[13]
        if name != "-":
            names.append(name)

print("read %d names" % len(names))

tree = SiteTree()


def extract_coords(s: str) -> tuple[str, str]:
    match = re.match(r"(\d{4})\w(\d{3})", s)
    return match.group(1), match.group(2)


def form_path(s: str, sky_part) -> str:
    lon, lat = extract_coords(s)
    return f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{sky_part}/tractor/{lon[:3]}/tractor-{s}.fits"


async def process_fits(url: str, name: str):
    file_name = f"data/{name}.fits"
    await tree.download(url, file_name)
    with fits.open(file_name, mode='update') as hdul:
        header = hdul[0].header
        data = hdul[1].data
        if not isinstance(data, FITS_rec):
            raise ValueError(f"Fits file {file_name} must be a table")
        header['url'] = url
        hdul.flush()


async def task(urls):
    tasks = [process_fits(url, name) for url, name in zip(urls, names)]
    tasks = [asyncio.create_task(task) for task in tasks]
    await asyncio.wait(tasks)

urls = [form_path(x, 'south') for x in names]

asyncio.run(task(urls))
