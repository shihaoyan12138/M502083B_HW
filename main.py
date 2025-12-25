import argparse
from pathlib import Path
import sys
import shutil
import uuid
import math

from utils.pdf_utils import read_pdf_text
from models.text_encoder import TextEncoder
from db.paper_db import PaperVectorDB
from models.image_encoder import ImageEncoder
from utils.image_utils import list_images
from db.image_db import ImageVectorDB


FILES_ROOT = Path("data/files")
UNKNOWN_DIR = FILES_ROOT / "unknown"
IMAGES_ROOT = Path("data/images")

FILES_ROOT.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def add_paper(args):
    pdf_path = Path(args.paper_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    encoder = TextEncoder()
    db = PaperVectorDB()

    text = read_pdf_text(str(pdf_path))
    embedding = encoder.encode(text)

    paper_id = str(uuid.uuid4())
    db.add_paper(
        paper_id,
        text[:2000],
        embedding,
        metadata={"filename": pdf_path.name}
    )

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        topics = ["unknown"]

    for topic in topics:
        target = FILES_ROOT / topic
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_path, target / pdf_path.name)

    print(f"Paper added: {pdf_path.name} → {topics}")


def search_paper(args):
    encoder = TextEncoder()
    db = PaperVectorDB()

    if db.collection.count() == 0:
        print("⚠️ Paper database is empty. Please add papers first.")
        return

    query_emb = encoder.encode(args.query)
    results = db.search(query_emb, top_k=3)

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not metadatas:
        print("⚠️ No matching papers found.")
        return

    # 1. distance -> similarity（可能为负）
    similarities = [1 - d for d in distances]

    # 2. Softmax with temperature
    max_sim = max(similarities)
    exp_scores = [math.exp((s - max_sim) / 0.5) for s in similarities]
    total = sum(exp_scores)

    percent_scores = [s / total * 100 for s in exp_scores]

    # 3. 输出
    print("🔍 Paper Search Results (Top-3):")
    for i, (meta, pct) in enumerate(zip(metadatas, percent_scores), start=1):
        print(f"{i}. {meta['filename']} | relevance={pct:.2f}%")


def batch_classify(args):
    encoder = TextEncoder()
    db = PaperVectorDB()

    unknown_pdfs = list(UNKNOWN_DIR.glob("*.pdf"))
    if not unknown_pdfs:
        print("No files in unknown/")
        return

    # 已存在的类别（排除 unknown）
    categories = [
        d for d in FILES_ROOT.iterdir()
        if d.is_dir() and d.name != "unknown"
    ]

    if not categories:
        print("No existing categories to classify into.")
        return

    print(f"Classifying {len(unknown_pdfs)} papers...")

    for pdf in unknown_pdfs:
        text = read_pdf_text(str(pdf))
        emb = encoder.encode(text)

        # 在已有论文向量中找最相似的
        results = db.search(emb, top_k=1)
        best_meta = results["metadatas"][0][0]

        # 从 filename 推断所属类别
        target_category = None
        for cat in categories:
            if (cat / best_meta["filename"]).exists():
                target_category = cat
                break

        if target_category is None:
            print(f"⚠️ Skip {pdf.name}, no matching category.")
            continue

        shutil.move(pdf, target_category / pdf.name)
        print(f"➡️ {pdf.name} → {target_category.name}")

    print("✅ Batch classification completed.")


import math

def search_image(args):
    encoder = ImageEncoder()
    db = ImageVectorDB()

    images = list_images(IMAGES_ROOT)
    if not images:
        print("⚠️ No images found in data/images/.")
        return

    # 1. 如果数据库为空，先建立图像索引（只做一次）
    if db.collection.count() == 0:
        print("📦 Building image vector database...")
        for img in images:
            emb = encoder.encode_image(str(img))
            db.add_image(
                image_id=str(uuid.uuid4()),
                embedding=emb,
                metadata={
                    "filename": img.name,
                    "path": str(img)
                }
            )

    # 2. 编码文本查询
    query_emb = encoder.encode_text(args.query)

    # 3. 搜索 Top-3
    results = db.search(query_emb, top_k=3)

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not metadatas:
        print("⚠️ No matching images found.")
        return

    # 4. distance -> similarity（可能为负）
    similarities = [1 - d for d in distances]

    # 5. Temperature Softmax（控制分布形状）
    T = 0.5
    max_sim = max(similarities)
    exp_scores = [math.exp((s - max_sim) / T) for s in similarities]
    total = sum(exp_scores)

    percent_scores = [s / total * 100 for s in exp_scores]

    # 6. 输出结果
    print("🖼️ Image Search Results (Top-3):")
    for i, (meta, pct) in enumerate(zip(metadatas, percent_scores), start=1):
        print(f"{i}. {meta['filename']} | relevance={pct:.2f}%")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Local Multimodal AI Agent"
    )

    subparsers = parser.add_subparsers(dest="command")

    # add_paper
    p_add = subparsers.add_parser("add_paper")
    p_add.add_argument("paper_path", type=str)
    p_add.add_argument("--topics", type=str, default="")
    p_add.set_defaults(func=add_paper)

    # search_paper
    p_search = subparsers.add_parser("search_paper")
    p_search.add_argument("query", type=str)
    p_search.set_defaults(func=search_paper)

    # batch_classify
    p_batch = subparsers.add_parser("batch_classify")
    p_batch.set_defaults(func=batch_classify)

    # search_image
    p_img = subparsers.add_parser("search_image")
    p_img.add_argument("query", type=str)
    p_img.set_defaults(func=search_image)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
