"""Import PDFs and images and run the local, sequential OCR workflow."""
import fcntl
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .config import load_config, resolve_config
from .local import apply_model_selection, local_config, run_local
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
        if index.is_file():
            name = json.loads(index.read_text(encoding="utf-8")).get("batch")
            if isinstance(name, str) and name not in ("", "..") and Path(name).name == name:
                candidate = root / name
                if candidate.is_dir() and not candidate.is_symlink() and not (candidate / "production/summary.json").exists():
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
            with atomic_output(index) as temporary:
                temporary.write_text(json_text({"batch": output.name, "source": str(source)}), encoding="utf-8")
        with batch_lock(output / ".run.lock"):
            try:
                yield output, resumed, signatures
            except InputChangedError:
                index.unlink()
                raise


def render_pages(source, output, settings, progress=None, *, reuse_existing=False, signatures=None):
    import pypdfium2 as pdfium

    if signatures is None:
        excluded = {output.parent.resolve(), Path(settings["output_root"]).resolve()}
        files, base = input_files(source, settings, excluded)
    else:
        files = list(signatures)
        base = source if source.is_dir() else source.parent
    dpi = render_dpi(settings)

    def pages():
        for number, file in enumerate(files, 1):
            if signatures is not None:
                check_input(file, signatures[file])
            doc_id = file.relative_to(base).as_posix()
            if progress:
                progress(f"导入文件 {number}/{len(files)} · {doc_id}")
            else:
                print(f"导入 {doc_id}", flush=True)
            if file.suffix.lower() != ".pdf":
                from PIL import Image

                image = Path("pages") / doc_id / ("1" + file.suffix.lower())
                if not (reuse_existing and (output / image).is_file()):
                    with Image.open(file) as source_image:
                        source_image.verify()
                    with atomic_output(output / image) as temporary:
                        shutil.copyfile(file, temporary)
                if signatures is not None:
                    check_input(file, signatures[file])
                yield {"doc_id": doc_id, "page_id": 1, "image": image.as_posix(), "split": "mine"}
                continue
            try:
                with pdfium.PdfDocument(file) as doc:
                    page_count = len(doc)
                    for index in range(page_count):
                        image = Path("pages") / doc_id / f"{index + 1}.png"
                        cached = reuse_existing and (output / image).is_file()
                        if progress:
                            label = "复用 PDF 页图" if cached else "PDF"
                            progress(f"{label} · 第 {index + 1}/{page_count} 页 · "
                                     f"文件 {number}/{len(files)} · {doc_id}")
                        if not cached:
                            page = doc[index]
                            try:
                                bitmap = page.render(scale=dpi / 72)
                                try:
                                    with atomic_output(output / image) as temporary:
                                        bitmap.to_pil().save(temporary, format="PNG")
                                finally:
                                    bitmap.close()
                            finally:
                                page.close()
                        if signatures is not None:
                            check_input(file, signatures[file])
                        yield {"doc_id": doc_id, "page_id": index + 1, "image": image.as_posix(), "split": "mine"}
            except pdfium.PdfiumError as error:
                raise ValueError(f"无法读取 PDF {file}：{error}") from error

    manifest = output / "pages.jsonl"
    write_rows(manifest, pages())
    return manifest


def process_inputs(source, settings_path, config_path, out, import_only):
    settings = load_config(settings_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (out or Path(settings["output_root"]) / stamp).resolve()
    if source.is_dir() and source.resolve().is_relative_to(Path(settings["output_root"]).resolve()):
        raise ValueError("请将输入文件夹与结果目录分开")
    output.mkdir(parents=True, exist_ok=True)
    with batch_lock(output / ".run.lock"):
        run_batch(source, settings, config_path, output, import_only)


def runtime_config(settings, config_path):
    config = local_config(settings, config_path)
    apply_model_selection(config, config["mining"])
    return resolve_config(config, config_path.parent)


def run_batch(source, settings, config_path, output, import_only):
    if (output / "production" / "summary.json").exists():
        raise ValueError("该批次已完成，请使用新的输出目录")
    pages = render_pages(source, output / "input", settings)
    print(f"页面清单 {pages}", flush=True)
    if import_only:
        return
    config = runtime_config(settings, config_path)
    run = output / "production"
    run_local(config, output, [(output / "capture" / "regions.jsonl", run)], pages)
    print(f"复核批次 {run}", flush=True)
