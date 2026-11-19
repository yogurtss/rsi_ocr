import re
import unicodedata
from dataclasses import dataclass
from html import escape, unescape
from html.parser import HTMLParser

try:
    from rapidfuzz.distance.Levenshtein import distance as edit_distance
except ImportError:
    def edit_distance(left, right):
        previous = list(range(len(right) + 1))
        for i, a in enumerate(left, 1):
            row = [i]
            for j, b in enumerate(right, 1):
                row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
            previous = row
        return previous[-1]


OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")
NUMBER = re.compile(r"[+\-−]?(?:\d+(?:[.,:/-]\d+)*|[.,]\d+)(?:[eE][+\-−]?\d+)?[%％‰]?")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def similarity(left, right):
    return 1 - edit_distance(left, right) / max(len(left), len(right), 1)


def unfence(text):
    match = re.fullmatch(r"```[a-zA-Z]*[ \t]*\n(.*?)\n```", text.strip(), flags=re.S)
    return match[1].strip() if match else text.strip()


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    height: int
    width: int
    text: str

    @property
    def position(self):
        return self.row, self.col, self.height, self.width


class HtmlTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.in_row = False
        self.state = "before"
        self.section = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table" and self.state == "before":
            self.state = "table"
        elif tag in {"thead", "tbody", "tfoot", "caption"} and self.state == "table" and not self.in_row and self.section is None:
            self.section = tag
        elif tag == "tr" and self.state == "table" and not self.in_row and self.section != "caption":
            self.in_row = True
            self.rows.append([])
        elif tag in {"td", "th"} and self.in_row and self.cell is None:
            attrs = dict(attrs)
            height, width = int(attrs.get("rowspan", 1)), int(attrs.get("colspan", 1))
            if not (1 <= height <= 512 and 1 <= width <= 512):
                raise ValueError("invalid cell span")
            self.cell = (tag, height, width, [])
        elif tag == "br" and self.cell:
            self.cell[3].append("\n")
        elif tag not in {"b", "i", "em", "strong", "span", "sub", "sup", "code", "u", "s", "p"} or not self.cell:
            raise ValueError(f"unexpected HTML table tag <{tag}>")
        elif tag in {"sub", "sup"}:
            self.cell[3].append(f"<{tag}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag != "br":
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.cell and tag == self.cell[0]:
            _, height, width, parts = self.cell
            self.rows[-1].append((height, width, "".join(parts).strip()))
            self.cell = None
        elif tag == "tr" and self.in_row and self.cell is None:
            self.in_row = False
        elif tag == self.section and not self.in_row:
            self.section = None
        elif tag == "table" and self.state == "table" and not self.in_row and self.section is None:
            self.state = "after"
        elif self.cell and tag in {"sub", "sup"}:
            self.cell[3].append(f"</{tag}>")
        elif not self.cell or tag in {"table", "tr", "td", "th", "caption", "thead", "tbody", "tfoot"}:
            raise ValueError(f"unexpected HTML table end tag </{tag}>")

    def handle_data(self, text):
        if self.cell:
            self.cell[3].append(text)
        elif text.strip() and self.section != "caption":
            raise ValueError("text outside table cells")


def parse_html(text):
    parser = HtmlTable()
    parser.feed(text)
    parser.close()
    if parser.state != "after" or not parser.rows:
        raise ValueError("expected one closed HTML table")
    occupied = set()
    cells = []
    for row, entries in enumerate(parser.rows):
        col = 0
        for height, width, content in entries:
            while (row, col) in occupied:
                col += 1
            if row + height > len(parser.rows):
                raise ValueError("rowspan extends beyond the table")
            region = {(r, c) for r in range(row, row + height) for c in range(col, col + width)}
            if region & occupied:
                raise ValueError("overlapping table cells")
            occupied.update(region)
            cells.append(Cell(row, col, height, width, content))
            col += width
        if not any(r == row for r, _ in occupied):
            raise ValueError("empty table row")
    return cells


def parse_otsl(text):
    tokens = list(OTSL.finditer(text))
    if not tokens or text[:tokens[0].start()].strip():
        raise ValueError("OTSL must begin with a cell token")
    grid = [[]]
    for i, token in enumerate(tokens):
        end = tokens[i + 1].start() if i + 1 < len(tokens) else len(text)
        kind = token[1]
        raw_value = text[token.end():end].strip()
        if re.search(r"<[A-Za-z]+cel>", raw_value):
            raise ValueError("unknown OTSL cell token")
        value = unescape(raw_value)
        if kind != "fcel" and value:
            raise ValueError("unexpected content in OTSL continuation")
        if kind == "nl":
            if not grid[-1]:
                raise ValueError("empty OTSL row")
            grid.append([])
        else:
            grid[-1].append((kind, value))
    if not grid[-1]:
        grid.pop()
    positions = {(r, c): value for r, row in enumerate(grid) for c, value in enumerate(row)}
    expected = {}
    cells = []
    for (row, col), (tag, value) in positions.items():
        if tag not in {"fcel", "ecel"}:
            continue
        width = height = 1
        while positions.get((row, col + width), (None,))[0] == "lcel":
            width += 1
        while positions.get((row + height, col), (None,))[0] == "ucel":
            height += 1
        cells.append(Cell(row, col, height, width, value))
        for dy in range(height):
            for dx in range(width):
                if dx or dy:
                    expected[row + dy, col + dx] = "xcel" if dx and dy else "lcel" if dx else "ucel"
    actual = {pos: tag for pos, (tag, _) in positions.items() if tag in {"xcel", "lcel", "ucel"}}
    if not cells or actual != expected:
        raise ValueError("inconsistent OTSL merged cells")
    return cells


def table_cells(text, source="auto"):
    text = unfence(text)
    if source == "otsl" or (source == "auto" and OTSL.match(text)):
        return parse_otsl(text)
    return parse_html(text)


def table_html(cells):
    rows = [[] for _ in range(max(cell.row + cell.height for cell in cells))]
    for cell in cells:
        attrs = ""
        if cell.height > 1:
            attrs += f' rowspan="{cell.height}"'
        if cell.width > 1:
            attrs += f' colspan="{cell.width}"'
        content = escape(cell.text, quote=False).replace("\n", "<br>")
        for tag in ("sub", "sup"):
            content = content.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        rows[cell.row].append(f"<td{attrs}>{content}</td>")
    return "<table>" + "".join("<tr>" + "".join(row) + "</tr>" for row in rows) + "</table>"


def table_otsl(cells):
    grid = [{} for _ in range(max(cell.row + cell.height for cell in cells))]
    for cell in cells:
        content = escape(cell.text, quote=False)
        for tag in ("sub", "sup"):
            content = content.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        grid[cell.row][cell.col] = "<fcel>" + content if content else "<ecel>"
        for dy in range(cell.height):
            for dx in range(cell.width):
                if dx or dy:
                    grid[cell.row + dy][cell.col + dx] = "<xcel>" if dx and dy else "<lcel>" if dx else "<ucel>"
    return "".join("".join(row.get(i, "<ecel>") for i in range(max(row) + 1)) + "<nl>" for row in grid)


def table_metrics(predicted, reference):
    left = {cell.position: normalize(cell.text) for cell in predicted}
    right = {cell.position: normalize(cell.text) for cell in reference}
    shared = left.keys() & right.keys()
    denominator = len(left) + len(right)
    structure = 2 * len(shared) / denominator
    content = 2 * sum(similarity(left[key], right[key]) for key in shared) / denominator
    numeric_keys = {key for key in left.keys() | right.keys()
                    if NUMBER.search(left.get(key, "")) or NUMBER.search(right.get(key, ""))}
    numbers = (sum(key in shared and NUMBER.findall(left[key]) == NUMBER.findall(right[key])
                   for key in numeric_keys) / len(numeric_keys)) if numeric_keys else 1.0
    return structure, content, numbers
