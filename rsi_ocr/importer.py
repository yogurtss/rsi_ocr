"""Import PDFs and images and run the local, sequential OCR workflow."""
import fcntl
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_config
from .local import apply_model_selection, local_config, run_local
from .store import atomic_output, write_rows


def render_pages(source, output, settings, progress=None):
    import pypdfium2 as pdfium

    extensions = {ext.lower() for ext in settings["extensions"]}
    if not extensions or extensions - {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("extensions 可选 .pdf、.png、.jpg 和 .jpeg")
    if not source.exists():
        raise ValueError(f"输入路径不存在：{source}")
    if source.is_dir():
        files = source.rglob("*") if settings["recursive"] else source.iterdir()
        excluded = {output.parent.resolve(), Path(settings["output_root"]).resolve()}
        files = sorted(p for p in files if p.is_file() and p.suffix.lower() in extensions
                       and not any(p.resolve().is_relative_to(root) for root in excluded))
        base = source
    else:
        files = [source] if source.suffix.lower() in extensions else []
        base = source.parent
    if not files:
        raise ValueError(f"没有匹配类型的文件：{source}")
    dpi = settings["dpi"]
    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi 必须是正整数")

    def pages():
        for number, file in enumerate(files, 1):
            doc_id = file.relative_to(base).as_posix()
            if progress:
                progress(f"导入文件 {number}/{len(files)} · {doc_id}")
            else:
                print(f"导入 {doc_id}", flush=True)
            if file.suffix.lower() != ".pdf":
                from PIL import Image

                with Image.open(file) as source_image:
                    source_image.verify()
                image = Path("pages") / doc_id / ("1" + file.suffix.lower())
                with atomic_output(output / image) as temporary:
                    shutil.copyfile(file, temporary)
                yield {"doc_id": doc_id, "page_id": 1, "image": image.as_posix(), "split": "mine"}
                continue
            try:
                with pdfium.PdfDocument(file) as doc:
                    page_count = len(doc)
                    for index in range(page_count):
                        if progress:
                            progress(f"PDF · 第 {index + 1}/{page_count} 页 · "
                                     f"文件 {number}/{len(files)} · {doc_id}")
                        page = doc[index]
                        image = Path("pages") / doc_id / f"{index + 1}.png"
                        try:
                            bitmap = page.render(scale=dpi / 72)
                            try:
                                with atomic_output(output / image) as temporary:
                                    bitmap.to_pil().save(temporary, format="PNG")
                            finally:
                                bitmap.close()
                        finally:
                            page.close()
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
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"批次正在生成：{output}") from error
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
