# MLFflow logging utilities.
from pathlib import Path
import base64
import html
from contextlib import redirect_stdout
from io import StringIO

from PIL import Image
import io
from urllib.parse import quote

import mlflow
import logging
from pytorch_lightning.loggers import MLFlowLogger, Logger


def get_mlflow_logger(trainer) -> MLFlowLogger:
    """Extract the MLFlow logger from the trainer, if it is being used."""
    logger = trainer.logger
    if isinstance(logger, MLFlowLogger):
        return logger

    if isinstance(logger, Logger):
        return None

    for logger in getattr(trainer, "loggers", []) or []:
        if isinstance(logger, MLFlowLogger):
            return logger

    return None


def build_html(
    mlflow_logger: Logger,
    plots_dir: Path,
    gallery_dir: Path,
    arti_dir: Path,
) -> Path:
    """Makes an index html file in mlflow and adds the plots in plots_dir to it."""
    plots_dir = plots_dir.resolve()
    section_prefix = str(arti_dir.relative_to(gallery_dir))
    sections = get_image_paths(plots_dir, section_prefix)

    # Check if index file already exists.
    index_exists = check_index_exists(mlflow_logger, gallery_dir)
    html_page = get_html_page(mlflow_logger, gallery_dir, index_exists)

    if sections:
        for section, images in sections:
            html_page += write_html_section(mlflow_logger, section, images, arti_dir)

    return "\n".join(html_page)


def check_index_exists(mlflow_logger: Logger, gallery_dir: Path) -> bool:
    """Checks if the index.html already exists at the given gallery_dir.

    If it exists, append to the gallery path instead of rewriting the file.
    """
    run_id = mlflow_logger.run_id
    arti_files = mlflow_logger.experiment.list_artifacts(run_id, path=str(gallery_dir))
    index_path = gallery_dir / 'index.html'
    index_file_exists = any([file.path == str(index_path) for file in arti_files])

    return index_file_exists


def get_html_page(
    mlflow_logger: Logger, gallery_dir: Path, index_exists: bool
) -> list[str]:
    """Get read or instantiate the html page based on whether it already exists.

    If index.html already exists in gallery_dir, then import it and append the plots
    in plots_dir to this index.html.
    Otherwise, generate a new index.html file and write the plots to it.
    """
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    if index_exists:
        run = mlflow_logger.experiment.get_run(mlflow_logger.run_id)
        artifact_uri = run.info.artifact_uri.rstrip("/")
        index_path = (gallery_dir / 'index.html').as_posix()
        index_path = f"{artifact_uri}/{index_path}"
        with redirect_stdout(StringIO()):
            html_page = mlflow.artifacts.load_text(index_path)
        html_page = html_page.splitlines()
    else:
        html_page = []
        html_page += generate_html_header()

    return html_page


def get_image_paths(plots_dir: Path, section_prefix: str):
    """Get paths to images contained in a root directory, grouped by subfolder."""
    groups = {}
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    for img_path in plots_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
            rel = img_path.relative_to(plots_dir)
            section_name = section_prefix
            section_name += rel.parts[0] if len(rel.parts) > 1 else "_root"
            groups.setdefault(section_name, []).append(img_path)

    for section in groups.keys():
        groups[section] = sorted(groups[section])

    return sorted(groups.items())


def generate_html_header():
    """Generate the header for the html file."""
    html_header = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        f"<title>PLOTS</title>",
        "<style>",
        "body{font:14px/1.4 system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:20px}",
        "h1{font-size:20px;margin:0 0 12px}",
        "details{margin:12px 0;border:1px solid #ddd;border-radius:8px;padding:8px}",
        "summary{cursor:pointer;font-weight:600}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:8px;margin-top:8px}",
        ".card{border:1px solid #eee;border-radius:8px;padding:6px;overflow:hidden;background:#fff}",
        ".card img {width:100%;height:300px;object-fit:contain;display:block;cursor:pointer}",
        ".caption{font-size:12px;margin-top:4px;word-break:break-all;color:#444}",
        "#lightbox{position:fixed;inset:0;background:rgba(0,0,0,0.85);display:none;align-items:center;justify-content:center;z-index:10000}",
        "#lightbox.open{display:flex}",
        "#lightbox img{max-width:96vw;max-height:92vh;box-shadow:0 0 15px #000;border-radius:4px}",
        "#lightbox-close{position:absolute;top:12px;right:20px;color:white;font-size:28px;cursor:pointer;font-weight:bold}",
        "</style>",

        "<script>",
        "function openLightbox(src){",
        "  const lb = document.getElementById('lightbox');",
        "  document.getElementById('lightbox-img').src = src;",
        "  lb.classList.add('open');",
        "}",
        "function closeLightbox(){",
        "  document.getElementById('lightbox').classList.remove('open');",
        "}",
        "document.addEventListener('keydown', function(e){",
        "  if(e.key === 'Escape') closeLightbox();",
        "});",
        "</script>",

        # Lightbox markup
        "<div id='lightbox' onclick='closeLightbox()'>",
        "  <span id='lightbox-close' onclick='closeLightbox()'>&times;</span>",
        "  <img id='lightbox-img' src='' onclick='event.stopPropagation()'>",
        "</div>",

        f"<h1>PLOTS</h1>",
    ]

    return html_header


def write_html_section(
     mlflow_logger: Logger, section: str, image_paths: list[Path, ...], arti_dir: Path
) -> list[str, ...]:
    """Write a section of the index html file generated in build_html.

    The section is made up of thumbnails that link to the original image in the MLFlow
    artifact directory structure. Clicking the thumbnail opens the original image in
    a new tab.
    """
    html_section = []
    html_section.append(
        f"<details><summary>{html.escape(section)} ({len(image_paths)})</summary>"
        )
    html_section.append("<div class='grid'>")
    for img_path in image_paths:
        caption = img_path.stem
        cap = html.escape(caption)
        thumb_src = html.escape(generate_thumbnail(img_path))

        html_section.append(
            "<div class='card'>"
            f"<img loading='lazy' src='{thumb_src}' alt='{cap}' "
            f"onclick='openLightbox(this.src)'>"
            f"<div class='caption'>{cap}</div>"
            "</div>"
        )

    html_section.append("</div></details>")

    return html_section


def generate_thumbnail(path: Path, max_size: int = 1024, quality: int = 30) -> str:
    """Generate the thumbnail that goes into the gallery."""
    ext = path.suffix.lower()
    img = Image.open(path)

    # Create a small thumbnail in-place
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Prepare buffer
    buf = io.BytesIO()

    if ext in {".jpg", ".jpeg"}:
        # Ensure RGB (JPEG doesn't support transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        mime = "image/jpeg"
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    elif ext in {".png", ".gif", ".webp"}:
        mime = {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}[ext]
        img.save(buf, format=img.format or "PNG", optimize=True)
    else:
        mime = "image/png"
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGBA")
        img.save(buf, format="PNG", optimize=True)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"
