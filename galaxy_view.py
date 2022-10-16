from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox


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
        "unprocessed": "Unprocessed",
        "good": "Good",
        "bad": "Bad"
    }
    def __init__(self, vm: GalaxyViewViewModel):
        self.vm = vm
        self._fig: Figure = plt.figure()
        self._fig.subplots_adjust(bottom=0.02)

        self._ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3, fig=self._fig)
        self._ax_main.get_xaxis().set_visible(False)
        self._ax_main.get_yaxis().set_visible(False)

        self._ax_sub = plt.subplot2grid((4, 3), (0, 2), fig=self._fig)
        self._ax_sub.yaxis.set_label_position("right")
        self._ax_sub.yaxis.tick_right()

        ax_checkbox = plt.subplot2grid((4, 3), (1, 2), fig=self._fig, aspect='equal')
        show_keys = self._additional_plot.keys()
        show_values = self._additional_plot.values()
        self.checkbox_btns = CheckButtons(ax_checkbox, show_values, [k in vm.show for k in show_keys])

        ax_display = plt.subplot2grid((4, 3), (2, 2), fig=self._fig, aspect='equal')
        display_keys = list(self._display_variants.keys())
        display_values = self._display_variants.values()
        self.display_variants = RadioButtons(ax_display, display_values, display_keys.index(vm.selection_main))

        ax_radio = plt.subplot2grid((24, 6), (18, 0), colspan=2, rowspan=4, fig=self._fig, aspect='equal')
        ax_radio.axis('off')
        sample_keys = list(self._smples.keys())
        self.radio_btns = RadioButtons(ax_radio, self._smples.values(), sample_keys.index(vm.selection_sample))

        ax_points_title = plt.subplot2grid((24, 6), (18, 2), colspan=8, fig=self._fig)
        TextBox(ax_points_title, "", initial="Points opacity", textalignment="center", color='w')
        ax_points_title.axis("off")
        ax_slider_points = plt.subplot2grid((24, 6), (19, 2), colspan=8, fig=self._fig)
        self.slider_points = Slider(ax_slider_points, "", valmin=0, valmax=1, valinit=vm.opacity_points)

        ax_points_title = plt.subplot2grid((24, 6), (20, 2), colspan=8, fig=self._fig)
        TextBox(ax_points_title, "", initial="Mask opacity", textalignment="center", color='w')
        ax_points_title.axis("off")
        ax_slider_mask = plt.subplot2grid((24, 6), (21, 2), colspan=8, fig=self._fig)
        self.slider_mask = Slider(ax_slider_mask, "", valmin=0, valmax=1, valinit=vm.opacity_mask)

        ax_prev = plt.subplot2grid((24, 7), (22, 0), rowspan=2, fig=self._fig)
        self.btn_prev = Button(ax_prev, "<--")

        ax_processed = plt.subplot2grid((24, 7), (22, 1), rowspan=2, colspan=2, fig=self._fig)
        self.btn_processed = Button(ax_processed, "Set processed")

        ax_ref = plt.subplot2grid((24, 7), (22, 3), rowspan=2, fig=self._fig)
        self.btn_ref = Button(ax_ref, "ref")

        ax_good = plt.subplot2grid((24, 7), (22, 4), colspan=2, fig=self._fig)
        self.btn_good = Button(ax_good, "Set good")

        ax_bad = plt.subplot2grid((24, 7), (23, 4), colspan=2, fig=self._fig)
        self.btn_bad = Button(ax_bad, "Set bad")

        ax_next = plt.subplot2grid((24, 7), (22, 6), rowspan=2, fig=self._fig)
        self.btn_next = Button(ax_next, "-->")


    def show(self):
        plt.show()


class GalaxyViewViewModel:
    def __init__(self,
                 show: list[str] = None,
                 opacity_points: float = 0.3, opacity_mask: float = 0.3,
                 selection_main: str = "picture", selection_sample: str = "unprocessed"):

        if selection_main not in GalaxyView._display_variants.keys():
            raise ValueError(f"selection_main nust be one of {list(GalaxyView._display_variants.keys())}")
        if selection_sample not in GalaxyView._smples.keys():
            raise ValueError(f"selection_sample nust be one of {list(GalaxyView._smples.keys())}")


        self.show = show or ['peaks']
        self.opacity_points = opacity_points
        self.opacity_mask = opacity_mask
        self.selection_main = selection_main
        self.selection_sample = selection_sample


if __name__ == "__main__":
    GalaxyViewViewModel()
    gv = GalaxyView(GalaxyViewViewModel())
    gv.show()

