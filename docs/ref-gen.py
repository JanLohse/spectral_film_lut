from pathlib import Path

PROJECT_NAME = "spectral_film_lut"
SPECIAL_CAPS = {
    "GUI",
    "CSS",
    "BW",
    "EXR",
    "CA",
    "II",
    "160S",
    "160C",
    "III",
    "PDII",
    "DPII",
    "3513DI",
    "3523XD",
    "FP100C",
    "100F",
    "100D",
    "IV",
    "XY",
    "LUT",
}  # always uppercase

TAG_MAPPING = {
    "Color Science": [
        "color_space",
        "densiometry",
        "film_spectral",
        "grain_generation",
        "xy_lut",
    ],
    "Data": [
        "color_space",
        "css_theme",
        "file_formats",
        "film_data",
        "wratten_filters",
        "densiometry",
    ],
    "UI": ["css_theme", "film_loader", "filmstock_selector", "gui", "splash_screen"],
    "Util": [
        "utils",
        "splash_screen",
        "film_loader",
        "gui_objects",
        "grain_generation",
        "xy_lut",
    ],
    "Module": ["init", "index"],
    "Film Stock": ["negative", "print", "reversal"],
}
ALTERNATE_TAGS = {"Data", "Film Stock"}  # if no other tags are matched

TAG_TO_ICON = [
    ("Color Science", "palette"),
    ("Film Stock", "film"),
    ("UI", "app-window"),
    ("Data", "database"),
    ("Util", "tool-case"),
    ("Reference", "puzzle"),
]

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


def get_tags(name: str) -> list[str]:
    tags = {"Reference"}  # always present
    name_lower = name.lower()

    found_any = False

    for tag, keywords in TAG_MAPPING.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                tags.add(tag)
                found_any = True
                break  # avoid duplicate checks for same tag

    if not found_any:
        tags |= ALTERNATE_TAGS

    return sorted(tags)


def get_icon(all_names: str, tags: list[str]):
    if "reference" in all_names:
        return "icon: lucide/book-open-text\n"

    for tag, icon in TAG_TO_ICON:
        if tag in tags:
            return f"icon: lucide/{icon}\n"

    return ""


def write_file(py_file: Path):
    module, out_file, display_name = get_paths(py_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_names = py_file.name + out_file.name + display_name.lower()

    tags = get_tags(all_names)

    tag_block = "\n".join(f"  - {t}" for t in tags)

    icon_block = get_icon(all_names, tags)

    content = (
        "---\n"
        f"{icon_block}"
        "tags:\n"
        f"{tag_block}\n"
        "---\n\n"
        f"# {display_name}\n\n"
        f"::: {module}\n"
    )

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
