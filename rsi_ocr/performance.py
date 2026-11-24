"""Batch-scoped vLLM concurrency samples and line plots."""
from time import monotonic

from PIL import Image, ImageDraw

from .store import atomic_output


class PerformanceRecord:
    def __init__(self, models):
        self.started = monotonic()
        self.limits = {name: model["max_sequences"] for name, model in models.items() if model["enabled"]}
        self.limits["layout"] = 1
        self.samples = []
        self.recording = True
        self.image = None

    def sample(self, values):
        self.samples.append((round(monotonic() - self.started, 3), values))

    def snapshot(self):
        return {"limits": self.limits, "samples": self.samples[-120:], "recording": self.recording,
                "image": self.image}

    def save(self, path):
        width = 1100
        image = Image.new("RGB", (width, 90 + 145 * len(self.limits)), "white")
        draw = ImageDraw.Draw(image)
        draw.text((26, 15), "Model concurrency  /  running requests and layout calls", fill="#172135", font_size=23)
        end = max(0.001, self.samples[-1][0])
        for i, (name, limit) in enumerate(self.limits.items()):
            top = 77 + 145 * i
            draw.text((26, top - 24), f"{name.upper()}   /   limit {limit}", fill="#172135", font_size=17)
            peak = scale(self.samples, name, limit)
            for value in sorted({0, peak // 2, peak}):
                y = top + 96 - round(96 * value / peak)
                draw.line((60, y, width - 30, y), fill="#dae0e8")
                draw.text((26, y - 7), str(value), fill="#45536a", font_size=13)
            plot(draw, self.samples, name, (60, top, width - 30, top + 96), (0, end), peak,
                 ("#167782", "#355fbb", "#8549a5", "#ae6015")[i % 4], 2)
        y = image.height - 27
        for fraction in (0, .25, .5, .75, 1):
            x = 60 + fraction * (width - 90)
            draw.text((x, y), f"{end * fraction:.0f}s", anchor="mt", fill="#45536a", font_size=13)
        with atomic_output(path) as temporary:
            image.save(temporary, format="PNG")


def scale(samples, name, limit):
    return max(1, limit, max((values.get(name) or 0 for _, values in samples), default=0))


def plot(draw, samples, name, bounds, interval, peak, color, weight):
    left, top, right, bottom = bounds
    start, end = interval
    previous = None
    for timestamp, values in samples:
        value = values.get(name)
        if value is None:
            previous = None
            continue
        point = (left + round((right - left) * (timestamp - start) / max(.001, end - start)),
                 bottom - round((bottom - top) * value / peak))
        if previous is None:
            draw.point(point, fill=color)
        else:
            draw.line((previous, point), fill=color, width=weight)
        previous = point


def terminal_line(samples, name, limit, width, rows):
    """Rasterize the same time series into 2-by-4 Braille cells."""
    image = Image.new("1", (2 * width, 4 * rows))
    plot(ImageDraw.Draw(image), samples, name, (0, 0, image.width - 1, image.height - 1),
         (samples[0][0], samples[-1][0]), scale(samples, name, limit), 1, 1)
    bits = ((1, 2, 4, 64), (8, 16, 32, 128))
    lines = []
    for y in range(0, image.height, 4):
        cells = []
        for x in range(0, image.width, 2):
            mask = sum(bit for dx, column in enumerate(bits) for dy, bit in enumerate(column)
                       if image.getpixel((x + dx, y + dy)))
            cells.append(chr(0x2800 + mask) if mask else " ")
        lines.append("".join(cells))
    return lines
