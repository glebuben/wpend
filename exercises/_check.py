"""Мини-проверялка для упражнений: печатает PASS/FAIL и объясняет каждую проверку.

Проверки здесь -- оракулы, а не сравнение с эталонным выводом: каждая опирается
на утверждение, верное по причинам вне твоего кода (сохраняющаяся величина,
порядок схемы, совпадение двух независимых путей вычисления).  Поэтому
"прошло" здесь действительно значит "правильно", а не "как вчера".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # чтобы работал import wpend


class Checks:
    def __init__(self, title: str):
        self.title = title
        self.failed = 0
        self.skipped = 0
        print()
        print("=" * 74)
        print("  " + title)
        print("=" * 74)

    def expect(self, name: str, ok: bool, why: str = "", hint: str = ""):
        print(("  [PASS] " if ok else "  [FAIL] ") + name)
        if why:
            print("         " + why)
        if not ok:
            self.failed += 1
            if hint:
                print("         подсказка: " + hint)
        return ok

    def skip(self, name: str, reason: str = ""):
        self.skipped += 1
        print("  [ -- ] " + name + (f"   ({reason})" if reason else ""))

    def done(self) -> bool:
        print("-" * 74)
        if self.failed:
            print(f"  не прошло проверок: {self.failed}")
        else:
            tail = f", пропущено {self.skipped}" if self.skipped else ""
            print("  всё сошлось" + tail)
        print()
        return self.failed == 0
