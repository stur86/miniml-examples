"""Build the GitHub Pages site: export each example in examples.yaml to static
HTML, without code, and write an index page that links to them.

Usage:
    uv run python scripts/build_site.py [--config examples.yaml] [--out _site]
"""

import argparse
import html
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="wrap">
  <p class="eyebrow">marimo notebooks</p>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
</header>
<main class="wrap">
  <ul class="grid">
{items}
  </ul>
</main>
<footer>
  <div class="wrap">
    <a href="{repo}">Source on GitHub</a>
  </div>
</footer>
</body>
</html>
"""

CARD_TEMPLATE = """    <li class="card">
      <h2><a href="{slug}.html">{name}</a></h2>
      <p>{description}</p>
      <div class="links">
        <span class="open">Open notebook &rarr;</span>
        <a class="source" href="{source}">source</a>
      </div>
    </li>"""


def load_config(path: Path) -> dict:
    """Read and check the examples configuration."""
    config = yaml.safe_load(path.read_text())
    examples = config.get("examples") or []
    slugs = set()
    for ex in examples:
        for key in ("name", "path", "slug"):
            if not ex.get(key):
                sys.exit(f"Example {ex!r} is missing '{key}'")
        if ex["slug"] in slugs or ex["slug"] == "index":
            sys.exit(f"Slug '{ex['slug']}' is used twice or is reserved")
        slugs.add(ex["slug"])
        if not (ROOT / ex["path"]).is_file():
            sys.exit(f"Notebook not found: {ex['path']}")
    return config


def export(notebook: Path, target: Path) -> None:
    """Run a notebook and export it as static HTML, without its code."""
    print(f"Exporting {notebook.relative_to(ROOT)} -> {target}", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html",
            "--no-include-code", "--force",
            str(notebook), "-o", str(target),
        ],
        check=True,
        cwd=ROOT,
    )


def write_index(config: dict, out: Path) -> None:
    """Write the index page that links to every example, and its stylesheet."""
    repo = config.get("repo", "").rstrip("/")
    items = [
        CARD_TEMPLATE.format(
            slug=html.escape(ex["slug"]),
            name=html.escape(ex["name"]),
            description=html.escape(ex.get("description", "")),
            source=html.escape(f"{repo}/blob/master/{ex['path']}"),
        )
        for ex in config["examples"]
    ]
    (out / "index.html").write_text(
        INDEX_TEMPLATE.format(
            title=html.escape(config.get("title", "Examples")),
            subtitle=html.escape(config.get("subtitle", "")),
            repo=html.escape(repo),
            items="\n".join(items),
        )
    )
    shutil.copyfile(ROOT / "theme" / "site.css", out / "style.css")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "examples.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for ex in config["examples"]:
        export(ROOT / ex["path"], args.out / f"{ex['slug']}.html")
    write_index(config, args.out)
    # Serve the files as they are, without Jekyll processing
    (args.out / ".nojekyll").touch()
    print(f"Site written to {args.out}")


if __name__ == "__main__":
    main()
