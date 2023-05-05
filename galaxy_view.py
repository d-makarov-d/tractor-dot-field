from __future__ import annotations

import asyncio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import time
import queue
import matplotlib.image as mpimg
import copy
from PIL import Image
from os import path
import argparse

from web import SiteTree
from db.brick_item import BrickDB, BrickItem
from app_preferences import AppPreferences

callback_queue = queue.Queue()


def _from_flux(flux: np.ndarray) -> np.ndarray:
    return 22.5 - 2.5 * np.log10(flux)


class DbIterator:
    def __init__(self, db: BrickDB):
        self._status_map_indices = dict(map(lambda x: (x, 0), BrickItem.STATUSES))
        self._db = db

        self._update_statuses()

    def _update_statuses(self):
        self._status_to_id = dict(map(lambda x: (x, []), self._status_map_indices.keys()))
        id_status = self._db.con.cursor().execute("SELECT id, status FROM peaks").fetchall()
        for id, status in id_status:
            self._status_to_id[status] = self._status_to_id[status] + [id]

    def next_by_status(self, step: int, status: str, tree: SiteTree) -> tuple[GalaxyViewViewModel, int, int]:
        n_items = len(self._status_to_id[status])
        if n_items == 0:
            raise IndexError("%s is empty" % status)

        if self._status_map_indices[status] + step < 0:
            idx = 0
        else:
            idx = (self._status_map_indices[status] + step) % n_items
        self._status_map_indices[status] = idx
        item = self._db.find(BrickItem, "id == '%s'" % self._status_to_id[status][idx])[0]

        return GalaxyViewViewModel(tree, item), idx, n_items

    def set_status(self, item: BrickItem, status: str):
        if status not in BrickItem.STATUSES:
            raise ValueError("Expected status in %s, git %s" % (BrickItem.STATUSES, status))
        new_item = copy.copy(item)
        new_item.status = status
        self._db.update(new_item)
        self._update_statuses()


class GalaxyView:
    _additional_plot = {
        "peaks": "Show peaks",
        "masks": "Show masks",
        "points": "Show points",
    }
    _display_variants = {
        "picture": "Picture",
        "field": "Field"
    }
    _smples = {
        BrickItem.STATUSES[0]: "Unprocessed",
        BrickItem.STATUSES[1]: "Good",
        BrickItem.STATUSES[2]: "Bad"
    }

    def __init__(self, db_iter: DbIterator, tree: SiteTree):
        self._tree = tree
        self._db_iterator = db_iter
        plt.ion()

        self._fig: Figure = plt.figure()
        self._fig.subplots_adjust(bottom=0.02)

        self._ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3, fig=self._fig)
        self._ax_main.get_xaxis().set_visible(False)
        self._ax_main.get_yaxis().set_visible(False)
        self.vm = None
        try:
            self.vm, i, n_items = db_iter.next_by_status(0, BrickItem.STATUSES[0], tree)
        except IndexError:
            self._ax_main.set_title("Empty")

        self._ax_sub = plt.subplot2grid((4, 3), (0, 2), fig=self._fig)
        self._ax_sub.yaxis.set_label_position("right")
        self._ax_sub.yaxis.tick_right()

        ax_checkbox = plt.subplot2grid((4, 3), (1, 2), fig=self._fig, aspect='equal')
        show_keys = self._additional_plot.keys()
        show_values = self._additional_plot.values()
        show = self.vm.show if self.vm is not None else []
        self.checkbox_btns = CheckButtons(ax_checkbox, list(show_values), [k in show for k in show_keys])
        self.checkbox_btns.on_clicked(self.redraw)

        ax_display = plt.subplot2grid((4, 3), (2, 2), fig=self._fig, aspect='equal')
        display_keys = list(self._display_variants.keys())
        display_values = self._display_variants.values()
        selection_main = self.vm.selection_main if self.vm is not None else "picture"
        self.display_variants = RadioButtons(ax_display, list(display_values), display_keys.index(selection_main))
        self.display_variants.on_clicked(self.redraw)

        ax_radio = plt.subplot2grid((24, 6), (18, 0), colspan=2, rowspan=4, fig=self._fig, aspect='equal')
        ax_radio.axis('off')
        sample_keys = list(self._smples.keys())
        selection = sample_keys.index(self.vm.selection_sample) if self.vm is not None else None
        self.radio_btns = RadioButtons(ax_radio, list(self._smples.values()), selection)
        self.radio_btns.on_clicked(self._on_type_selected)

        ax_points_title = plt.subplot2grid((24, 6), (18, 2), colspan=8, fig=self._fig)
        TextBox(ax_points_title, "", initial="Points opacity", textalignment="center", color='w')
        ax_points_title.axis("off")
        ax_slider_points = plt.subplot2grid((24, 6), (19, 2), colspan=8, fig=self._fig)
        opacity_points = self.vm.opacity_points if self.vm is not None else 0.3
        self.slider_points = Slider(ax_slider_points, "", valmin=0, valmax=1, valinit=opacity_points)
        self.slider_points.on_changed(self.redraw)

        ax_points_title = plt.subplot2grid((24, 6), (20, 2), colspan=8, fig=self._fig)
        TextBox(ax_points_title, "", initial="Mask opacity", textalignment="center", color='w')
        ax_points_title.axis("off")
        ax_slider_mask = plt.subplot2grid((24, 6), (21, 2), colspan=8, fig=self._fig)
        opacity_mask = self.vm.opacity_mask if self.vm is not None else 0.3
        self.slider_mask = Slider(ax_slider_mask, "", valmin=0, valmax=1, valinit=opacity_mask)
        self.slider_mask.on_changed(self.redraw)

        ax_prev = plt.subplot2grid((24, 7), (22, 0), rowspan=2, fig=self._fig)
        self.btn_prev = Button(ax_prev, "<--")
        self.btn_prev.on_clicked(self._step_back)

        ax_unprocessed = plt.subplot2grid((24, 7), (22, 1), rowspan=2, colspan=2, fig=self._fig)
        self.btn_unprocessed = Button(ax_unprocessed, "Set unprocessed")
        self.btn_unprocessed.on_clicked(lambda _: self._set_status("unprocessed"))

        ax_ref = plt.subplot2grid((24, 7), (22, 3), rowspan=2, fig=self._fig)
        self.btn_ref = Button(ax_ref, "ref")
        self.btn_ref.on_clicked(lambda _: self._on_ref_required())

        ax_good = plt.subplot2grid((24, 7), (22, 4), colspan=2, fig=self._fig)
        self.btn_good = Button(ax_good, "Set good")
        self.btn_good.on_clicked(lambda _: self._set_status("good"))

        ax_bad = plt.subplot2grid((24, 7), (23, 4), colspan=2, fig=self._fig)
        self.btn_bad = Button(ax_bad, "Set bad")
        self.btn_bad.on_clicked(lambda _: self._set_status("bad"))

        ax_next = plt.subplot2grid((24, 7), (22, 6), rowspan=2, fig=self._fig)
        self.btn_next = Button(ax_next, "-->")
        self.btn_next.on_clicked(self._step_forward)

        self._show_picture()

        self._pic_download = 0.0
        self._pic = None

        self._update_by_status(BrickItem.STATUSES[0], 0)

    def show(self):
        plt.show()

    def redraw(self, _=None):
        for child in self._ax_main.get_children():
            try:
                child.remove()
            except NotImplementedError:
                pass
        max_x = self.vm.width if self._pic is None else self._pic.shape[1]
        max_y = self.vm.height if self._pic is None else self._pic.shape[0]
        if self.display_variants.value_selected == self._display_variants["picture"]:
            if self._pic is None:
                self._ax_main.text(0.5, 0.5, "%.2f" % self._pic_download, c='r')
            else:
                resized = Image.fromarray(self._pic).resize((1500, 1500), Image.Resampling.LANCZOS)
                self._ax_main.imshow(resized, extent=[0, max_x, 0, max_y])
        elif self.display_variants.value_selected == self._display_variants["field"]:
            self._ax_main.imshow(self.vm.Z, extent=[0, max_x, 0, max_y])

        draw_peaks, draw_masks, draw_points = self.checkbox_btns.get_status()
        if draw_peaks:
            r = (self.vm.item._area / np.pi) ** 0.5
            c = plt.Circle((self.vm.item._peak_x, self.vm.item._peak_y), radius=r, linewidth=1, fill=None, color='r')
            self._ax_main.add_patch(c)
        if draw_points:
            self._ax_main.scatter(self.vm.x, self.vm.y, alpha=self.slider_points.val, c='g', marker='2')
        if draw_masks:
            masked_img = np.ma.masked_where(self.vm.item._mask_area == False,
                                            0.7 * np.ones_like(self.vm.item._mask_area))
            self._ax_main.imshow(masked_img, extent=[0, max_x, 0, max_y], alpha=self.slider_mask.val,
                                 cmap=plt.get_cmap('Reds'), clim=[0, 1])

        sub_x = _from_flux(self.vm.item.flux_g) - _from_flux(self.vm.item.flux_r)
        sub_y = _from_flux(self.vm.item.flux_r)
        self._ax_sub.cla()
        self._ax_sub.invert_yaxis()
        self._ax_sub.scatter(sub_x, sub_y)

    def _show_picture(self):
        pool = ThreadPoolExecutor(2)
        dt = 1.0 / 60
        self.__t = time.process_time()
        curr_vm = self.vm

        def progress_cb(p):
            def f():
                self._pic_download = p
                self.redraw()

            if time.process_time() - self.__t > dt and self.vm is curr_vm:
                self.__t = time.process_time()
                callback_queue.put(f)

        async def task():
            self._pic = await self.vm.picture(progress_cb)
            if self._pic is not None and self.vm is curr_vm:
                callback_queue.put(lambda: self.redraw())

        picture_task = pool.submit(lambda: asyncio.run(task()))
        pool.submit(picture_task)

    def _on_type_selected(self, _):
        status = list(self._smples.keys())[list(self._smples.values()).index(self.radio_btns.value_selected)]
        self._update_by_status(status, 0)

    def _step_forward(self, _):
        status = list(self._smples.keys())[list(self._smples.values()).index(self.radio_btns.value_selected)]
        self._update_by_status(status, 1)

    def _step_back(self, _):
        status = list(self._smples.keys())[list(self._smples.values()).index(self.radio_btns.value_selected)]
        self._update_by_status(status, -1)

    def _set_status(self, status: str):
        self._db_iterator.set_status(self.vm.item, status)
        status = list(self._smples.keys())[list(self._smples.values()).index(self.radio_btns.value_selected)]
        self._update_by_status(status, 0)

    def _update_by_status(self, status: str, step: int):
        if self.vm is None:
            return

        max_x = self.vm.x.max() if self._pic is None else self._pic.shape[1]
        max_y = self.vm.y.max() if self._pic is None else self._pic.shape[0]
        self._ax_main.set_xlim([0, max_x])
        self._ax_main.set_ylim([0, max_y])
        self._pic = None
        try:
            self.vm, i, n_items = self._db_iterator.next_by_status(step, status, self._tree)
            self._ax_main.set_title("%i/%i %s" % (i, n_items, self.vm.item._brick_name))
            self._show_picture()
        except IndexError:
            self._ax_main.clear()
            self._ax_main.set_title("Empty")

    def _on_ref_required(self):
        ra, dec = self.vm.item._ra, self.vm.item._dec
        print(f"https://www.legacysurvey.org/viewer/?ra={ra}&dec={dec}&zoom=12&layer=ls-dr9")


class GalaxyViewViewModel:
    def __init__(self,
                 tree: SiteTree,
                 item: BrickItem,
                 show: list[str] = None,
                 opacity_points: float = 0.3, opacity_mask: float = 0.3,
                 selection_main: str = "picture", selection_sample: str = "unprocessed"):

        if selection_main not in GalaxyView._display_variants.keys():
            raise ValueError(f"selection_main nust be one of {list(GalaxyView._display_variants.keys())}")
        if selection_sample not in GalaxyView._smples.keys():
            raise ValueError(f"selection_sample nust be one of {list(GalaxyView._smples.keys())}")

        self._tree = tree

        self.show = show or ['peaks']
        self.opacity_points = opacity_points
        self.opacity_mask = opacity_mask
        self.selection_main = selection_main
        self.selection_sample = selection_sample

        self.x, self.y, self.Z = item._x, item._y, item._Z
        self.width = item.width
        self.height = item.height
        brick_name = item._brick_name
        self.item = item

        # download image of the sky
        self._picture: np.ndarray = None
        self._img_path = re.sub(
            r"/tractor/.+/tractor-(\d{3})(\d[pm]\d+).+$",
            r"/coadd/\g<1>/\g<1>\g<2>/legacysurvey-\g<1>\g<2>-image.jpg",
            item._url
        )
        self._img_file = re.sub(r"^(.+\.).+$", r"\g<1>jpg", brick_name)

    async def picture(self, progress: Callable[[float], None] = None) -> np.ndarray:
        if not path.isfile(self._img_file):
            res = await self._tree.download(self._img_path, self._img_file, progress)
            if res is None:
                return None
        img = mpimg.imread(self._img_file)
        # remove(self._img_file)
        return img


def _run():
    tree = SiteTree()
    db = BrickDB("main_db.sqlite")
    db_iterator = DbIterator(db)
    unique_names = db.con.cursor().execute("SELECT DISTINCT brick_name FROM peaks").fetchall()

    gv = GalaxyView(db_iterator, tree)
    while True:
        try:
            callback = callback_queue.get(False)  # doesn't block
            callback()
            plt.pause(1e-5)
        except queue.Empty:  # raised when queue is empty
            plt.pause(1e-5)

    tree.close()


def run(args: list[str], name: str, prefs: AppPreferences):
    parser = argparse.ArgumentParser(description="View processing results")
    parser.usage = parser.format_usage().replace('usage: %s' % args[0], '%s %s' % (args[0], name))
    parser.parse_args(args[2:])
    _run()


if __name__ == "__main__":
    _run()
