# Make an html gallery of a folder of html images, including all images in subfolders.

from pathlib import Path
import base64
import html


def build_html(root_dir: Path, title: str = "Plots Gallery") -> Path:
    """Makes an index html file in mlflow that displays all plots below root_dir."""
    root_dir = root_dir.resolve()
    sections = get_image_paths(root_dir)

    # HTML file text.
    html_page = []
    html_page += generate_html_header(title)

    if not sections:
        html_page.append("<p>No images found.</p>")
    else:
        for section, images in sections:
            html_page += write_html_section(section, images)

    return "\n".join(html_page)


def get_image_paths(root_dir: Path):
    """Get paths to images contained in a root directory, grouped by subfolder."""
    groups = {}
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    for img_path in root_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
            rel = img_path.relative_to(root_dir)
            section = rel.parts[0] if len(rel.parts) > 1 else "_root"
            groups.setdefault(section, []).append(img_path)

    for section in groups.keys():
        groups[section] = sorted(groups[section])

    return sorted(groups.items())


def generate_html_header(title: str):
    """Generate the header for the html file."""
    html_header = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font:14px/1.4 system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:20px}",
        "h1{font-size:20px;margin:0 0 12px}",
        "details{margin:12px 0;border:1px solid #ddd;border-radius:8px;padding:8px}",
        "summary{cursor:pointer;font-weight:600}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin-top:8px}",
        ".card{border:1px solid #eee;border-radius:8px;padding:6px;overflow:hidden;background:#fff}",
        ".card a{display:block;text-decoration:none;color:inherit}",
        ".card img{width:100%;height:180px;object-fit:contain;display:block}",
        ".caption{font-size:12px;margin-top:4px;word-break:break-all;color:#444}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    return html_header


def write_html_section(section: str, image_paths: list[Path, ...]):
    """Write section of html file, with corresponding image paths to go in it."""
    html_section = []
    html_section.append(f"<details open><summary>{html.escape(section)} ({len(image_paths)})</summary>")
    html_section.append("<div class='grid'>")
    for img_path in image_paths:
        caption = img_path.stem
        # Link to the image; clicking opens just the image asset in MLflow
        html_section.append(
            "<div class='card'>"
            f"<img loading='lazy' src='{html.escape(data_uri(img_path))}' alt='{html.escape(caption)}'>"
            f"<div class='caption'>{html.escape(caption)}</div>"
            "</a>"
            "</div>"
        )
    html_section.append("</div></details>")

    return html_section


def data_uri(path: Path) -> str:
    """Get the mlflow compatible uri for given path."""
    mime = {
        ".png":"image/png",
        ".jpg":"image/jpeg",
        ".jpeg":"image/jpeg",
        ".gif":"image/gif",
        ".webp":"image/webp",
        ".svg":"image/svg+xml",
    }.get(path.suffix.lower(), "application/octet-stream")

    if path.suffix.lower() == ".svg":
        b64 = path.read_text(encoding="utf-8").encode("utf-8")
    else:
        b64 = path.read_bytes()

    return f"data:{mime};base64,{base64.b64encode(b64).decode('ascii')}"
