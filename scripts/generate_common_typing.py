from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TypedDictSpec:
    source_class: str
    output_class: str


TARGETS = (
    TypedDictSpec("BaseCameraConfig", "BaseCameraConfigKwargs"),
    TypedDictSpec("RobotConfig", "RobotConfigKwargs"),
)


class _QualifyCommonNames(ast.NodeTransformer):
    def __init__(self, common_class_names: set[str]) -> None:
        self.common_class_names = common_class_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.common_class_names:
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="common", ctx=ast.Load()), attr=node.id, ctx=ast.Load()),
                node,
            )
        return node


def _find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node

    msg = f"Could not find class {class_name!r} in stub file."
    raise ValueError(msg)


def _find_init_signature(class_node: ast.ClassDef) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node

    msg = f"Could not find __init__ for class {class_node.name!r}."
    raise ValueError(msg)


def _render_typeddict(
    spec: TypedDictSpec,
    module: ast.Module,
    qualifier: _QualifyCommonNames,
) -> list[str]:
    class_node = _find_class(module, spec.source_class)
    init_node = _find_init_signature(class_node)
    lines = [f"class {spec.output_class}(TypedDict, total=False):"]

    for arg in init_node.args.args[1:]:
        annotation = arg.annotation
        if annotation is None:
            annotation_text = "object"
        else:
            qualified_annotation = qualifier.visit(ast.fix_missing_locations(annotation))
            annotation_text = ast.unparse(qualified_annotation)
        lines.append(f"    {arg.arg}: {annotation_text}")

    if len(lines) == 1:
        lines.append("    pass")
    return lines


def generate_common_typing(stub_path: Path, output_path: Path) -> None:
    module = ast.parse(stub_path.read_text())
    common_class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    qualifier = _QualifyCommonNames(common_class_names)

    rendered_targets: list[str] = []
    for spec in TARGETS:
        rendered_targets.extend(_render_typeddict(spec, module, qualifier))
        rendered_targets.append("")

    output = "\n".join(
        [
            "# ATTENTION: auto generated from C++ stub files, use `make stubgen` to update!",
            '"""TypedDict helpers generated from `python/rcs/_core/common.pyi`."""',
            "from __future__ import annotations",
            "",
            "from typing import TypedDict",
            "",
            "from rcs._core import common",
            "",
            '__all__ = ["BaseCameraConfigKwargs", "RobotConfigKwargs"]',
            "",
            *rendered_targets[:-1],
        ]
    )
    output_path.write_text(f"{output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TypedDict helpers from common.pyi.")
    parser.add_argument(
        "--stub-path",
        default="python/rcs/_core/common.pyi",
        type=Path,
        help="Path to the generated common.pyi stub file.",
    )
    parser.add_argument(
        "--output-path",
        default="python/rcs/common_typing.py",
        type=Path,
        help="Path where the TypedDict helper module should be written.",
    )
    args = parser.parse_args()
    generate_common_typing(args.stub_path, args.output_path)


if __name__ == "__main__":
    main()
