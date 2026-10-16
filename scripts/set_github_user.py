"""Put your GitHub username into the README badge and the Colab notebook.

    python scripts/set_github_user.py <your-github-username>

Run it once, after you create the repository on GitHub and before you push.
Works on Windows, macOS and Linux (no sed needed).
"""
import sys
from pathlib import Path

PLACEHOLDER = "YOUR-USERNAME"
ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "README.md", *sorted((ROOT / "notebooks").glob("*.ipynb"))]


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit(__doc__)
    user = sys.argv[1].strip()
    changed = 0
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        if PLACEHOLDER in text:
            path.write_text(text.replace(PLACEHOLDER, user), encoding="utf-8")
            changed += 1
            print(f"updated {path.relative_to(ROOT)}")
    print(f"done ({changed} file(s) changed)" if changed else "nothing to change (already set?)")


if __name__ == "__main__":
    main()
