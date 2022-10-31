class Peak:
    def __init__(self, x: float, y: float, n_points: int, area:float, ra: float, dec: float):
        self.x = x
        self.y = y
        self.n_points = n_points
        self.area = area
        self.ra = ra
        self.dec = dec

    def __str__(self):
        return "Peak [ra: %.3f, dec: %.3f], points: %d, area: %.2f" % (self.ra, self.dec, self.n_points, self.area)