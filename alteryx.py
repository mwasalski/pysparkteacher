"""Turn Alteryx workflows, macros and packages into compact text for the model.

A saved workflow is largely canvas layout plus a cached copy of every tool's output schema.
Dropping those keeps big workflows inside the model's context window without touching the tool
configuration the translation depends on.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable

UPLOAD_TYPES = ["yxmd", "yxmc", "yxwz", "yxzp"]
_XML_SUFFIXES = (".yxmd", ".yxmc", ".yxwz")

# Comment boxes and containers: their styling is all layout. Other tools keep everything.
_CANVAS_PLUGINS = (
    "AlteryxGuiToolkit.TextBox.TextBox",
    "AlteryxGuiToolkit.ToolContainer.ToolContainer",
)
_CANVAS_STYLE = {"Font", "TextColor", "FillColor", "Shape", "Justification", "Style", "BG_Image"}
_TEXT_INPUT_ROWS = 20  # enough to show the shape of an inline table
_MAX_FILE_BYTES = 50 * 1024 * 1024  # per file unpacked from a .yxzp - guards against zip bombs
# Connection strings can carry a password inline, not only in <Passwords>.
_PASSWORD_RE = re.compile(r"(?i)(pwd|password)\s*=[^;|\"'&<]*")


def compact(document: str | bytes) -> str:
    """Strip what the translation does not need. Input that is not XML comes back as text."""
    if isinstance(document, str):
        document = document.strip()  # an XML declaration must be the very first thing
    try:
        root = ET.fromstring(document)
    except ET.ParseError:
        return document if isinstance(document, str) else document.decode("utf-8", "replace")

    for parent in list(root.iter()):
        _remove(parent, {"Passwords"})

    for node in root.iter("Node"):
        gui = node.find("GuiSettings")
        if gui is not None:
            _remove(gui, {"Position"})
        props = node.find("Properties")
        if props is None:
            continue
        # The schema each tool last produced: often most of the file. The workflow's own
        # <MetaInfo> (name, description) is not under a Node, so it stays.
        _remove(props, {"MetaInfo"})
        annotation = props.find("Annotation")
        if annotation is not None:
            _remove(annotation, {"DefaultAnnotationText", "Left"})  # the default repeats the config
            if not "".join(annotation.itertext()).strip():
                props.remove(annotation)
        config = props.find("Configuration")
        if config is not None and gui is not None and gui.get("Plugin") in _CANVAS_PLUGINS:
            _remove(config, _CANVAS_STYLE)

    for data in root.iter("Data"):  # Text Input rows, also the template inside Macro Input
        rows = data.findall("r")
        if len(rows) > _TEXT_INPUT_ROWS:
            for row in rows[_TEXT_INPUT_ROWS:]:
                data.remove(row)
            data.set("omitted_rows", str(len(rows) - _TEXT_INPUT_ROWS))

    ET.indent(root, space=" ")  # replaces the file's own, wider indentation
    return _PASSWORD_RE.sub(r"\1=***", ET.tostring(root, encoding="unicode"))


def read_uploads(files: Iterable[tuple[str, bytes]]) -> str:
    """Compact uploaded (name, content) pairs into one text, unpacking .yxzp packages.

    Each file is headed by an XML comment with its name, so the model can tell the workflow
    from the macros it calls.
    """
    documents: list[tuple[str, bytes]] = []
    for name, content in files:
        if name.lower().endswith(".yxzp"):
            documents.extend(_unpack(name, content))
        else:
            documents.append((name, content))
    return "\n\n".join(f"<!-- file: {name} -->\n{compact(content)}" for name, content in documents)


def _remove(parent: ET.Element, tags: set[str]) -> None:
    doomed = [child for child in parent if child.tag in tags]
    for child in doomed:
        parent.remove(child)
    if doomed and len(parent) == 0 and parent.text and not parent.text.strip():
        parent.text = None  # indentation that preceded the removed children


def _unpack(name: str, content: bytes) -> list[tuple[str, bytes]]:
    """A .yxzp is a zip of the workflow, its macros and its data files; keep the XML ones."""
    try:
        package = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{name} is not a valid .yxzp package") from exc

    documents = []
    with package:
        for member in package.infolist():
            if not member.filename.lower().endswith(_XML_SUFFIXES):
                continue
            with package.open(member) as f:
                data = f.read(_MAX_FILE_BYTES + 1)  # don't trust the size in the zip header
            if len(data) > _MAX_FILE_BYTES:
                raise ValueError(f"{member.filename} in {name} is over {_MAX_FILE_BYTES >> 20} MB")
            documents.append((f"{name}/{member.filename}", data))

    if not documents:
        raise ValueError(f"{name} contains no workflows or macros")
    return documents
