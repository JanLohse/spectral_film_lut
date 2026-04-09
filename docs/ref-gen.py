from pathlib import Path

PROJECT_NAME = "spectral_film_lut"
SPECIAL_CAPS = {"GUI", "CSS", "BW", "EXR", "CA"}  # always uppercase

SRC_DIR = Path("src") / PROJECT_NAME
OUT_DIR = Path("docs/reference")


def iter_modules():
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        yield py_file


def format_title(name: str) -> str:
    words = name.replace("_", " ").split()
    formatted = [
        w.upper() if w.upper() in SPECIAL_CAPS else w.capitalize() for w in words
    ]
    return " ".join(formatted)


def get_paths(py_file: Path):
    rel = py_file.relative_to(SRC_DIR)

    if py_file.name == "__init__.py":
        mod_parts = rel.parent.parts
        out_file = OUT_DIR / rel.parent / "index.md"
        name_source = rel.parent.name
    else:
        mod_parts = rel.with_suffix("").parts
        out_file = OUT_DIR / rel.with_suffix(".md")
        name_source = rel.stem

    module = ".".join((PROJECT_NAME, *mod_parts))

    # special case: project root
    if rel == Path("__init__.py"):
        display_name = "Reference"
    else:
        display_name = format_title(name_source)

    return module, out_file, display_name


def write_file(py_file: Path):
    module, out_file, display_name = get_paths(py_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    content = f"# {display_name}\n\n::: {module}\n"
    out_file.write_text(content, encoding="utf-8")


def clean_output():
    if OUT_DIR.exists():
        for f in OUT_DIR.rglob("*.md"):
            f.unlink()


def main():
    clean_output()

    for py_file in iter_modules():
        write_file(py_file)


if __name__ == "__main__":
    main()
