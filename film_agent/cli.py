"""CLI: film-agent [file|directory] --style ... --output ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import IMAGE_EXTENSIONS, STYLES, render_photo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Адаптивная обработка фотографий в стиле плёнки")
    parser.add_argument("source", type=Path, help="Фото или каталог фотографий")
    parser.add_argument("--style", choices=STYLES, default="hp5-120", help="Плёнка")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Файл или выходной каталог")
    parser.add_argument("--strength", type=float, default=1.0, help="Интенсивность стиля (0..2)")
    parser.add_argument("--grain", type=float, default=1.0, help="Интенсивность зерна (0..3)")
    parser.add_argument("--seed", type=int, default=42, help="Зерно воспроизводимо при одинаковом seed")
    parser.add_argument("--iso", type=int, help="Переопределить ISO из EXIF")
    parser.add_argument("--no-auto", action="store_true", help="Отключить адаптацию")
    parser.add_argument("--report", action="store_true", help="Сохранить отчёт JSON для каждого файла")
    args = parser.parse_args(argv)
    if not args.source.exists():
        parser.error(f"Источник не найден: {args.source}")
    if args.source.is_file():
        if args.output.is_dir():
            parser.error("Для одного фото --output должен указывать на имя выходного файла")
        pairs = [(args.source, args.output)]
    else:
        if args.output.suffix:
            parser.error("Для каталога --output должен быть каталогом")
        source = args.source.resolve()
        target_dir = args.output.resolve()
        files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.resolve().is_relative_to(target_dir)]
        pairs = [(p, args.output / p.relative_to(source).with_suffix(".jpg")) for p in files]
    if not pairs:
        print("Фотографий нет", file=sys.stderr)
        return 1
    errors = 0
    for src, dest in pairs:
        try:
            if src.resolve() == dest.resolve():
                raise ValueError("Выходной файл не должен совпадать с оригиналом")
            report = render_photo(src, dest, style=args.style, strength=args.strength,
                                  grain=args.grain, seed=args.seed, iso=args.iso, auto=not args.no_auto)
            if args.report:
                dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK: {src} → {dest} | EV {report['analysis']['exposure_ev']:+.2f}")
        except Exception as exc:
            print(f"ERROR: {src}: {exc}", file=sys.stderr)
            errors += 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
