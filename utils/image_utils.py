from pathlib import Path


def list_images(image_dir: str):
    """
    扫描目录下所有图片文件
    """
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
