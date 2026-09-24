"""`pdf-ocr` command.

    pdf-ocr analyse [PATH]   documents in input/ -> detections in output/
    pdf-ocr ingest  [PATH]   training material in training_data/sources/ -> training_data/unlabeled/
    pdf-ocr crawl SOURCE     download training material into .../sources/crawled/SOURCE/
    pdf-ocr circuit PATH     subcircuits from KiCad netlists (graph pattern matching)
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="PDF datasheet analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    analyse = commands.add_parser(
        "analyse", help="Render PDFs/images and run the YOLO schematic detector"
    )
    analyse.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "input",
        help="PDF, image, or folder of PDFs/images (default: input/)",
    )
    analyse.add_argument(
        "--output", "-o", type=Path, default=REPO_ROOT / "output",
        help="Output folder (default: output/)",
    )
    analyse.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    analyse.add_argument("--conf", type=float, help="Confidence threshold (default: from training config)")
    analyse.add_argument("--model", help="Path to model weights (default: from training config)")

    ingest = commands.add_parser(
        "ingest",
        help="Render training material from training_data/sources/ into training_data/unlabeled/",
    )
    ingest.add_argument(
        "sources",
        nargs="?",
        type=Path,
        help="Folder of PDFs/images, searched recursively (default: training_data/sources/)",
    )
    ingest.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    ingest.add_argument(
        "--dataset",
        choices=["pages", "subcircuits"],
        default="pages",
        help="Which training dataset's sources/ and unlabeled/ to use (default: pages)",
    )
    ingest.add_argument(
        "--max-pages", type=int, default=30, help="Pages rendered per PDF (default: 30, 0 = all)"
    )

    crawl = commands.add_parser(
        "crawl",
        help="Download training material (limits, licenses and hosts in training_project/config/crawl.yaml)",
    )
    crawl.add_argument("source", choices=["wikimedia", "wikimedia_blocks", "archive_org", "kicad_github", "urls"])
    crawl.add_argument("--limit", type=int, help="Max files this run (default: crawl.yaml per_run)")
    crawl.add_argument("--dry-run", action="store_true", help="Only list what would be downloaded")
    crawl.add_argument(
        "--refresh-netlists",
        action="store_true",
        help="kicad_github only: backfill netlists/schematic files for already crawled projects",
    )

    circuit = commands.add_parser(
        "circuit", help="Find subcircuits in KiCad netlists (.netlist.xml, .kicad_sch, or a folder)"
    )
    circuit.add_argument("path", type=Path, help="Netlist, .kicad_sch, or folder searched for *.netlist.xml")
    circuit.add_argument(
        "--output", "-o", type=Path, default=REPO_ROOT / "output" / "circuits",
        help="Output folder (default: output/circuits/)",
    )
    circuit.add_argument(
        "--gold", type=Path,
        default=REPO_ROOT / "training_project" / "training_data" / "circuits" / "gold",
        help="Gold set folder (default: training_data/circuits/gold/)",
    )
    circuit.add_argument("--init-gold", action="store_true", help="Write current matches as gold drafts to review")
    circuit.add_argument("--evaluate", action="store_true", help="Precision/recall against reviewed gold files")
    circuit.add_argument("--review", action="store_true",
                         help="Draw gold entries (or matches) into the schematic: output/circuits/review/")

    args = parser.parse_args()

    # Imported here so --help stays fast (these pull in torch/ultralytics)
    if args.command == "analyse":
        from pdf_ocr import pipeline

        pipeline.run(args.input, args.output, dpi=args.dpi, conf=args.conf, model_path=args.model)

    elif args.command == "ingest":
        from pdf_ocr import ingest as ingest_step
        from pdf_ocr.detect import training_paths

        sources_dir, unlabeled_dir = training_paths(args.dataset)
        summary = ingest_step.run(
            args.sources or sources_dir, unlabeled_dir, dpi=args.dpi, max_pages=args.max_pages
        )
        print(
            f"\nIngested {summary['ingested']} file(s) -> {summary['pages']} page(s) in {unlabeled_dir}"
            f"\nSkipped {summary['already_ingested']} already-ingested file(s)"
        )

    elif args.command == "circuit":
        from pdf_ocr.circuit import cli as circuit_cli

        circuit_cli.run(args.path, args.output, args.gold, evaluate=args.evaluate, init_gold=args.init_gold,
                        review=args.review)

    elif args.command == "crawl":
        from pdf_ocr import crawl as crawler

        if args.refresh_netlists and args.source != "kicad_github":
            parser.error("--refresh-netlists only works with kicad_github")
        ctx = crawler.run(
            args.source, limit=args.limit, dry_run=args.dry_run, refresh_netlists=args.refresh_netlists
        )
        print("\nSummary:")
        for key, count in sorted(ctx.summary.items()):
            print(f"  {key}: {count}")
        print(f"  disk used: {ctx.budget.used / 2**20:.0f} / {ctx.budget.max_bytes / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
