import math


class Bounds:
    def __init__(self, left, right, top, bottom):
        self.left = int(left)
        self.right = int(right)
        self.top = int(top)
        self.bottom = int(bottom)

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self):
        return Bounds(self.left, self.right, self.top, self.bottom)

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def subtract(self, other):
        return Bounds(self.left + other.left, self.right - other.right, self.top + other.top, self.bottom - other.bottom)

    def shift_x(self, x):
        self.left = self.left + x
        self.right = self.right + x

    def shift_y(self, y):
        self.top = self.top + y
        self.bottom = self.bottom + y

    def clamp_x(self, min_x, max_x):
        # Cut off the tile on the left
        if self.left < min_x:
            self.left = min_x

        # Cut off the tile on the right
        if self.right > max_x:
            self.right = max_x

    def clamp_y(self, min_y, max_y):
        # Cut off the tile on the top
        if self.top < min_y:
            self.top = min_y

        # Cut off the tile on the bottom
        if self.bottom > max_y:
            self.bottom = max_y

    def shift_x_bounds(self, min_x, max_x):
        extra = self.right - max_x

        # Shift the tile left
        if extra > 0:
            return -extra

        else:
            extra = min_x - self.left

            if extra > 0:
                # Shift the tile right
                return extra

        return 0

    def shift_y_bounds(self, min_y, max_y):
        extra = self.bottom - max_y

        # Shift the tile up
        if extra > 0:
            return -extra

        else:
            extra = min_y - self.top

            if extra > 0:
                # Shift the tile down
                return extra

        return 0

    def grow(self, other):
        self.left = self.left - other.left
        self.right = self.right + other.right
        self.top = self.top - other.top
        self.bottom = self.bottom + other.bottom

    def clamp(self):
        if self.left < 0:
            self.left = 0

        if self.right < 0:
            self.right = 0

        if self.top < 0:
            self.top = 0

        if self.bottom < 0:
            self.bottom = 0

    def print(self):
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    def to_dict(self):
        return {
            "x": self.left,
            "y": self.top,
            "width": self.right - self.left,
            "height": self.bottom - self.top,
        }


class Tile:
    def __init__(self, crop, mask, grow):
        self.crop = crop
        self.mask = mask
        self.grow = grow

    def __hash__(self):
        return hash((self.crop, self.mask, self.grow))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def has_size(self):
        width = self.mask.width() - self.grow.left - self.grow.right
        height = self.mask.height() - self.grow.top - self.grow.bottom
        return width > 0 and height > 0

    def clamp(self):
        self.mask.grow(self.grow)

        extra = self.crop.left - self.mask.left

        if extra > 0:
            self.mask.left = self.mask.left + extra
            self.grow.left = self.grow.left - extra

            if self.grow.left < 0:
                self.grow.left = 0

        extra = self.crop.top - self.mask.top

        if extra > 0:
            self.mask.top = self.mask.top + extra
            self.grow.top = self.grow.top - extra

            if self.grow.top < 0:
                self.grow.top = 0

        extra = self.mask.right - self.crop.right

        if extra > 0:
            self.mask.right = self.mask.right - extra
            self.grow.right = self.grow.right - extra

            if self.grow.right < 0:
                self.grow.right = 0

        extra = self.mask.bottom - self.crop.bottom

        if extra > 0:
            self.mask.bottom = self.mask.bottom - extra
            self.grow.bottom = self.grow.bottom - extra

            if self.grow.bottom < 0:
                self.grow.bottom = 0


def get_image_tiles(width, height, size, overlap):
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(size, int)
    assert isinstance(overlap, int)

    spacing = size - (overlap * 2)

    x_tiles = int(math.ceil(width / spacing))
    y_tiles = int(math.ceil(height / spacing))

    for x_tile in range(0, x_tiles):
        for y_tile in range(0, y_tiles):
            x = x_tile * spacing
            y = y_tile * spacing

            mask = Bounds(x, x + spacing, y, y + spacing)

            grow = Bounds(overlap, overlap, overlap, overlap)

            crop = mask.copy()
            crop.grow(grow)

            tile = Tile(crop, mask, grow)

            tile.crop.shift_x(tile.crop.shift_x_bounds(0, width))
            tile.crop.shift_y(tile.crop.shift_y_bounds(0, height))

            tile.crop.clamp_x(0, width)
            tile.crop.clamp_y(0, height)

            assert tile.crop.left >= 0
            assert tile.crop.right >= 0
            assert tile.crop.top >= 0
            assert tile.crop.bottom >= 0

            tile.clamp()

            assert tile.mask.left >= 0
            assert tile.mask.right >= 0
            assert tile.mask.top >= 0
            assert tile.mask.bottom >= 0

            assert tile.grow.left >= 0
            assert tile.grow.right >= 0
            assert tile.grow.top >= 0
            assert tile.grow.bottom >= 0

            assert isinstance(tile.crop.left, int)
            assert isinstance(tile.crop.right, int)
            assert isinstance(tile.crop.top, int)
            assert isinstance(tile.crop.bottom, int)

            assert isinstance(tile.mask.left, int)
            assert isinstance(tile.mask.right, int)
            assert isinstance(tile.mask.top, int)
            assert isinstance(tile.mask.bottom, int)

            assert isinstance(tile.grow.left, int)
            assert isinstance(tile.grow.right, int)
            assert isinstance(tile.grow.top, int)
            assert isinstance(tile.grow.bottom, int)

            if tile.has_size():
                yield tile


def test():
    for tile in set(get_image_tiles(1088, 1920, 1024, 16)):
        print(tile.crop.to_dict())
        print(tile.mask.to_dict())
        print(tile.grow.print())
        print("----")

#test()
