"""Self-contained HTML image galleries (thumbnails embedded as data URIs).

Moved out of ``src/evaluation/callbacks/utils/mlflow.py`` so scripts can build or
refresh galleries without importing mlflow, Lightning or torch. The mlflow module
re-exports every name, so existing imports keep working.
"""

import base64
import html
import io
from pathlib import Path

from PIL import Image


def build_gallery_html(plots_dir: Path, section_name: str) -> str:
    """Build a standalone HTML gallery containing every current image in a folder."""
    html_page = generate_gallery_header()
    html_page += build_gallery_section(plots_dir, section_name)
    return "\n".join(html_page)


def build_gallery_section(plots_dir: Path, section_name: str) -> list[str]:
    """Build one gallery section from every current image in ``plots_dir``."""
    return write_gallery_section(section_name, get_image_paths(plots_dir))


def get_image_paths(plots_dir: Path):
    """Get paths to images contained in a root directory, grouped by subfolder."""
    image_paths = []
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    for img_path in plots_dir.glob("*"):
        if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
            image_paths.append(img_path)

    image_paths = sorted(image_paths)

    return image_paths


def generate_gallery_header():
    """Generate the header for the html file."""
    html_header = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>PLOTS</title>",
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
        "<h1>PLOTS</h1>",
    ]

    return html_header


def write_gallery_section(section: str, image_paths: list[Path, ...]) -> list[str, ...]:
    """Write a section of the index html file generated in build_html.

    A section is made up of compressed small image thumbnail that expand when clicked.
    The quality is kept low to ensure that the html page is not too big in terms of
    memory. If the html page is larger than 50 Mb, mlflow does not display it.
    """
    html_section = []
    html_section.append(
        f"<details open><summary>{html.escape(section)} ({len(image_paths)})</summary>"
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


def generate_thumbnail(path: Path, max_size: int = 1024, quality: int = 90) -> str:
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
