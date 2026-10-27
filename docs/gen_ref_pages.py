"""Generate one API reference page per public module of ``src/womblex``.

Run by the ``gen-files`` plugin during ``mkdocs build``; nothing is written to
the work tree. A module or package whose name starts with an underscore is
internal and is left out.
"""

from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).parent.parent / "src"

nav = mkdocs_gen_files.Nav()

for path in sorted((SRC / "womblex").rglob("*.py")):
    parts = tuple(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if any(part.startswith("_") for part in parts):
        continue
    doc_path = Path("reference", *parts).with_suffix(".md")
    if path.name == "__init__.py":
        doc_path = Path("reference", *parts, "index.md")
    nav[parts] = doc_path.relative_to("reference").as_posix()
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(SRC.parent))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
