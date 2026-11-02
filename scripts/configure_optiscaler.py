#!/usr/bin/env python3
"""Apply release settings to the supplied OptiScaler INI, preserving its defaults/CRLF."""
import re
import sys
from pathlib import Path


def configure(source, settings):
    lines = source.split("\n")
    for setting in settings.splitlines():
        setting = setting.strip()
        if not setting or setting.startswith("#"):
            continue
        path, value = setting.split("=", 1)
        section, key = path.split(".", 1)
        starts = [i for i, line in enumerate(lines) if line.strip().casefold() == f"[{section}]".casefold()]
        if len(starts) != 1:
            raise ValueError(f"OptiScaler.ini must have exactly one [{section}]")
        start = starts[0]
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        matches = [i for i in range(start + 1, end)
                   if re.match(rf"\s*{re.escape(key)}\s*=", lines[i], re.IGNORECASE)]
        if len(matches) != 1:
            raise ValueError(f"OptiScaler.ini must have exactly one {key} in [{section}]")
        i = matches[0]
        lines[i] = f"{key}={value}" + ("\r" if lines[i].endswith("\r") else "")
    return "\n".join(lines)


def main():
    source, settings, output = map(Path, sys.argv[1:])
    try:
        # read_text() would translate CRLF before we could preserve it.
        with source.open(encoding="utf-8", newline="") as handle:
            result = configure(handle.read(), settings.read_text(encoding="utf-8"))
    except ValueError as error:
        sys.exit(str(error))
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
