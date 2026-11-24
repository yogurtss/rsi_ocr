"""Import PDFs and images and run the local, sequential OCR workflow."""
import fcntl
import json
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .config import load_config, resolve_config
from .store import atomic_output, digest, json_text, write_rows


def input_files(source, settings, excluded):
    extensions = {ext.lower() for ext in settings["extensions"]}
    if not extensions or extensions - {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("extensions 可选 .pdf、.png、.jpg 和 .jpeg")
    if not source.exists():
        raise ValueError(f"输入路径不存在：{source}")
    if source.is_dir():
        files = source.rglob("*") if settings["recursive"] else source.iterdir()
        files = sorted(p for p in files if p.is_file() and p.suffix.lower() in extensions
                       and not any(p.resolve().is_relative_to(root) for root in excluded))
        base = source
    else:
        files = [source] if source.suffix.lower() in extensions else []
        base = source.parent
    if not files:
        raise ValueError(f"没有匹配类型的文件：{source}")
    return files, base


@contextmanager
def batch_lock(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"批次正在生成：{path.parent}") from error
        yield


def file_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class InputChangedError(ValueError):
    pass


def check_input(path, signature):
    try:
        if file_signature(path) == signature:
            return
    except FileNotFoundError:
        pass
    raise InputChangedError(f"处理期间输入文件发生变化：{path}")


def render_dpi(settings):
    dpi = settings["dpi"]
    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi 必须是正整数")
    return dpi


@contextmanager
def resume_batch(root, source, settings, check_cancelled):
    dpi = render_dpi(settings)
    files, base = input_files(source, settings, {root})
    signatures = {}
    hashes = {}
    for path in files:
        signatures[path] = file_signature(path)
        content = sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_cancelled()
                content.update(block)
        check_input(path, signatures[path])
        hashes[path.relative_to(base).as_posix()] = content.hexdigest()
    key = digest(json_text(["pages-v1", str(source), dpi, hashes]).encode())
    index = root / ".resume" / f"{key}.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    with batch_lock(index.with_suffix(".lock")):
        check_cancelled()
        output = None
        previous = None
        if index.is_file():
            name = json.loads(index.read_text(encoding="utf-8")).get("batch")
            if isinstance(name, str) and name not in ("", "..") and Path(name).name == name:
                candidate = root / name
                if candidate.is_dir() and not candidate.is_symlink():
                    if (candidate / "production/summary.json").exists():
                        previous = candidate
                    else:
                        output = candidate
        resumed = output is not None
        if not resumed:
            while True:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                output = root / stamp
                try:
                    output.mkdir()
                except FileExistsError:
                    continue
                break
            if previous:
                with atomic_output(output / ".previous-batch") as temporary:
                    temporary.write_text(previous.name, encoding="utf-8")
            with atomic_output(index) as temporary:
                temporary.write_text(json_text({"batch": output.name, "source": str(source)}), encoding="utf-8")
        with batch_lock(output / ".run.lock"):
            try:
                yield output, resumed, signatures
            except InputChangedError:
                index.unlink()
                raise


@dataclass
class RenderPlan:
    manifest: Path
    rows: list
    base: Path
    dpi: int
    reuse_existing: bool
    signatures: dict
    counts: dict
    previous: Path | None

    def existing_image(self, row):
        image = self.manifest.parent / row["image"]
        if self.reuse_existing and image.is_file():
            return image
        if self.previous:
            previous = self.previous / row["image"]
            if previous.is_file():
                return previous
        return None

    def images(self, progress=None, check_cancelled=None):
        """Render in document order. Only the producer thread touches PDFium objects."""
        import pypdfium2 as pdfium

        current = None
        file_number = 0
        with ExitStack() as stack:
            for row in self.rows:
                if check_cancelled:
                    check_cancelled()
                doc_id = row["doc_id"]
                file = self.base / doc_id
                if doc_id != current:
                    stack.close()
                    current, doc = doc_id, None
                    file_number += 1
                    message = f"导入文件 {file_number}/{len(self.counts)} · {doc_id}"
                    if progress:
                        progress(message)
                    else:
                        print(message, flush=True)
                check_input(file, self.signatures[file])
                image = self.manifest.parent / row["image"]
                existing = self.existing_image(row)
                cached = existing is not None
                if cached and existing != image:
                    with atomic_output(image) as temporary:
                        shutil.copyfile(existing, temporary)
                number, total = row["page_id"], self.counts[doc_id]
                if file.suffix.lower() == ".pdf":
                    if progress and (not cached or number in (1, total) or number % 100 == 0):
                        label = "复用 PDF 页图" if cached else "PDF"
                        progress(f"{label} · 第 {number}/{total} 页 · 文件 {file_number}/{len(self.counts)} · {doc_id}")
                    if not cached:
                        try:
                            if doc is None:
                                doc = stack.enter_context(pdfium.PdfDocument(file))
                            with ExitStack() as page_resources:
                                page = doc[number - 1]
                                page_resources.callback(page.close)
                                bitmap = page.render(scale=self.dpi / 72)
                                page_resources.callback(bitmap.close)
                                with atomic_output(image) as temporary:
                                    bitmap.to_pil().save(temporary, format="PNG")
                                    check_input(file, self.signatures[file])
                        except pdfium.PdfiumError as error:
                            raise ValueError(f"无法读取 PDF {file}：{error}") from error
                elif not cached:
                    from PIL import Image

                    with Image.open(file) as source_image:
                        source_image.verify()
                    with atomic_output(image) as temporary:
                        shutil.copyfile(file, temporary)
                        check_input(file, self.signatures[file])
                check_input(file, self.signatures[file])
                yield row
        write_rows(self.manifest, self.rows)


def render_pages(source, output, settings, progress=None, *, reuse_existing=False, signatures=None,
                 streaming=False, check_cancelled=None):
    import pypdfium2 as pdfium

    indexed = signatures is not None
    if signatures is None:
        excluded = {output.parent.resolve(), Path(settings["output_root"]).resolve()}
        files, base = input_files(source, settings, excluded)
        signatures = {file: file_signature(file) for file in files}
    else:
        files = list(signatures)
        base = source if source.is_dir() else source.parent
    counts = {}
    rows = []
    dpi = render_dpi(settings)
    # A completed batch with the same input fingerprint can supply its rendered PNGs too.
    marker = output.parent / ".previous-batch"
    previous = None
    if indexed and marker.is_file():
        name = marker.read_text(encoding="utf-8").strip()
        candidate = output.parent.parent / name
        if name and name != ".." and Path(name).name == name and not candidate.is_symlink():
            previous = candidate / "input"
    for file in files:
        if check_cancelled:
            check_cancelled()
        check_input(file, signatures[file])
        doc_id = file.relative_to(base).as_posix()
        ext = file.suffix.lower()
        count = 1
        if ext == ".pdf":
            try:
                with pdfium.PdfDocument(file) as doc:
                    count = len(doc)
            except pdfium.PdfiumError as error:
                raise ValueError(f"无法读取 PDF {file}：{error}") from error
        counts[doc_id] = count
        for number in range(1, count + 1):
            image = Path("pages") / doc_id / f"{number}{'.png' if ext == '.pdf' else ext}"
            rows.append({"doc_id": doc_id, "page_id": number, "image": image.as_posix(), "split": "mine"})
    plan = RenderPlan(output / "pages.jsonl", rows, base, dpi, reuse_existing, signatures, counts, previous)
    if streaming:
        return plan
    for _ in plan.images(progress):
        pass
    return plan.manifest


def process_inputs(source, settings_path, config_path, out, import_only):
    settings = load_config(settings_path)
    source = source.resolve()
    if source.is_dir() and source.resolve().is_relative_to(Path(settings["output_root"]).resolve()):
        raise ValueError("请将输入文件夹与结果目录分开")
    if out is None:
        root = Path(settings["output_root"]).resolve()
        with resume_batch(root, source, settings, lambda: None) as (output, resumed, signatures):
            run_batch(source, settings, config_path, output, import_only, resumed=resumed, signatures=signatures)
        return
    output = out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with batch_lock(output / ".run.lock"):
        run_batch(source, settings, config_path, output, import_only)


def runtime_config(settings, config_path):
    from .local import apply_model_selection, local_config

    config = local_config(settings, config_path)
    apply_model_selection(config, config["mining"])
    return resolve_config(config, config_path.parent)


def run_batch(source, settings, config_path, output, import_only, *, resumed=False, signatures=None):
    if (output / "production" / "summary.json").exists():
        raise ValueError("该批次已完成，请使用新的输出目录")
    from .local import run_local

    pages = render_pages(source, output / "input", settings, reuse_existing=resumed, signatures=signatures,
                         streaming=not import_only)
    if import_only:
        print(f"页面清单 {pages}", flush=True)
        return
    config = runtime_config(settings, config_path)
    run = output / "production"
    run_local(config, output, [(output / "capture" / "regions.jsonl", run)], pages)
    print(f"复核批次 {run}", flush=True)
