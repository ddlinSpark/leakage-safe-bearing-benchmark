"""Check README links and Python command entry points."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_IMAGE = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\']', re.IGNORECASE)
HTTP_URL = re.compile(r"https?://[^\s)>\"']+")
PYTHON_COMMAND = re.compile(r"^\s*python(?:\.exe)?\s+(.+)$", re.IGNORECASE)


def check_file_target(root: Path, target: str) -> bool:
    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text or path_text.startswith(("http://", "https://", "mailto:")):
        return True
    return (root / path_text).resolve().exists()


def check_external(url: str) -> bool | None:
    try:
        request = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=10) as response:
            return response.status < 400
    except HTTPError as error:
        return error.code < 400
    except (URLError, TimeoutError, ValueError):
        return None


def command_lines(readme: str) -> list[str]:
    commands = []
    blocks = []
    in_block = False
    language = ""
    current = []
    for line in readme.splitlines():
        if not in_block and line.startswith("```"):
            in_block = True
            language = line[3:].strip().lower()
            current = []
        elif in_block and line.startswith("```"):
            if language in {"", "bash", "powershell"}:
                blocks.append("\n".join(current))
            in_block = False
            language = ""
            current = []
        elif in_block:
            current.append(line)
    for block in blocks:
        logical = ""
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.endswith("\\"):
                logical += stripped[:-1].strip() + " "
                continue
            logical += stripped
            if PYTHON_COMMAND.match(logical):
                commands.append(logical.strip())
            logical = ""
        if logical and PYTHON_COMMAND.match(logical):
            commands.append(logical.strip())
    return commands


def python_module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    for candidate in (
        ROOT / "src" / relative.with_suffix(".py"),
        ROOT / relative.with_suffix(".py"),
        ROOT / "src" / relative / "__init__.py",
        ROOT / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def inspect_python_command(command: str) -> dict[str, object]:
    tokens = shlex.split(command, posix=False)
    arguments = tokens[2:] if tokens[:2] == ["python", "-m"] else tokens[1:]
    entry = arguments[0] if arguments else ""
    if entry in {"-m", "pip"} or not entry:
        return {"command": command, "valid": True, "reason": "install command"}
    module_mode = tokens[1:2] == ["-m"]
    if module_mode:
        path = python_module_path(entry)
        options = arguments[1:]
    else:
        path = (ROOT / entry).resolve()
        options = arguments[1:]
    if path is None or not path.is_file():
        return {"command": command, "valid": False, "reason": f"missing {entry}"}
    requested = {
        value
        for value in options
        if value.startswith("--") and value not in {"--", "-h", "--help"}
    }
    completed = subprocess.run(
        [sys.executable, "-m", entry, "--help"]
        if module_mode
        else [sys.executable, str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    help_text = completed.stdout + completed.stderr
    missing = sorted(option for option in requested if option not in help_text)
    return {
        "command": command,
        "valid": completed.returncode == 0 and not missing,
        "reason": (
            "help command failed"
            if completed.returncode != 0
            else f"unknown options: {missing}"
            if missing
            else "entry point and options exist"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-external", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "reports" / "README_LINK_CHECK.md",
    )
    args = parser.parse_args()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = [
        *MARKDOWN_LINK.findall(readme),
        *HTML_IMAGE.findall(readme),
    ]
    local_targets = [
        target for target in targets if not target.startswith(("http://", "https://", "mailto:"))
    ]
    external_targets = [
        target.rstrip(".,;")
        for target in HTTP_URL.findall(readme)
    ]
    external_targets = sorted(set(external_targets))
    broken_local = sorted(
        target for target in local_targets if not check_file_target(ROOT, target)
    )
    external_results = {}
    if args.check_external:
        for target in sorted(set(external_targets)):
            external_results[target] = check_external(target)
    commands = [inspect_python_command(command) for command in command_lines(readme)]
    broken_commands = [command for command in commands if not command["valid"]]
    status = "PASS" if not broken_local and not broken_commands else "FAIL"

    lines = [
        "# README Link Check",
        "",
        "## Verdict",
        "",
        f"**{status}**",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total links checked | {len(local_targets) + len(external_targets)} |",
        f"| Valid local links | {len(local_targets) - len(broken_local)} |",
        f"| Broken local links | {len(broken_local)} |",
        f"| External links detected | {len(external_targets)} |",
        f"| Python commands checked | {len(commands)} |",
        f"| Broken Python commands | {len(broken_commands)} |",
        "",
        "## Broken local links",
        "",
    ]
    lines.extend(f"- `{target}`" for target in broken_local)
    if not broken_local:
        lines.append("- None.")
    lines.extend(["", "## Python commands", ""])
    for command in commands:
        marker = "PASS" if command["valid"] else "FAIL"
        lines.append(f"- **{marker}** `{command['command']}`: {command['reason']}")
    lines.extend(
        [
            "",
            "## External links",
            "",
            (
                "External links were checked with HTTP requests."
                if args.check_external
                else "External links were not verified automatically."
            ),
        ]
    )
    if args.check_external:
        for target in sorted(external_results):
            result = external_results[target]
            status_text = "PASS" if result is True else "FAIL" if result is False else "NOT VERIFIED"
            lines.append(f"- **{status_text}** {target}")
    elif external_targets:
        lines.extend(f"- {target}" for target in sorted(set(external_targets)))
    else:
        lines.append("- None.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"{status}: {len(targets)} links and {len(commands)} commands checked")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
