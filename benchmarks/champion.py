"""Kanonik sampiyon-skor kaydi. Dongunun sonu.

Sorun: longmemeval_eval.py skoru ekrana basip uctu, hicbir yere kaydetmedi.
Her instance bastan kosup ayni skoru yeniden kanitladi. Bu modul skoru tek
kanonik yere yazar: bir sonraki instance KOSMADAN once `best()` cagirir, skoru
gorur, koymaz.

Iki dosya:
  CHAMPION.json       en iyi (R@1) kosu, config + tarih + kaynak ile. Tek gercek.
  eval_ledger.jsonl   her kosu, append-only audit izi (immutable, Satoshi).

`update()` her kosudan sonra cagrilir; ledger'a her zaman ekler, CHAMPION'i
SADECE ayni soru-sayisinda R@1 gercekten gecilirse gunceller (gerileme
sampiyonu bozamaz).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent
CHAMPION = _DIR / "CHAMPION.json"
LEDGER = _DIR / "eval_ledger.jsonl"


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def best() -> dict[str, Any] | None:
    """Mevcut sampiyon. Kosmadan once BUNA bak. None ise hic kayit yok."""
    if not CHAMPION.exists():
        return None
    try:
        return json.loads(CHAMPION.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def update(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Bir kosu sonucunu kaydet. Returns: {'champion': bool, 'record': {...}}.

    result: {'n', 'R@1', 'R@5', 'R@10', 'by_type'?}  (evaluate_mnemonics ciktisi)
    meta:   {'config', 'ce_model'?, 'encoder'?, 'dataset'?, 'source'?}
    """
    if not result or result.get("R@1") is None:
        return {"champion": False, "record": None}

    record = {
        "n": result.get("n"),
        "R@1": result.get("R@1"),
        "R@5": result.get("R@5"),
        "R@10": result.get("R@10"),
        "by_type": result.get("by_type"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }

    # 1) Ledger'a her zaman ekle (append-only audit)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({k: v for k, v in record.items() if k != "by_type"}) + "\n")

    # 2) CHAMPION'i sadece ayni n'de R@1 gercekten gecilirse guncelle
    cur = best()
    is_champ = (
        cur is None
        or (record["n"] == cur.get("n") and record["R@1"] > cur.get("R@1", -1))
        or (record["n"] or 0) > (cur.get("n") or 0)  # daha buyuk n her zaman daha guvenilir
    )
    if is_champ:
        _atomic_write(CHAMPION, json.dumps(record, indent=2))
    return {"champion": is_champ, "record": record}


if __name__ == "__main__":
    c = best()
    if c:
        print(f"SAMPIYON: R@1={c['R@1']} R@5={c.get('R@5')} R@10={c.get('R@10')} "
              f"(n={c['n']}, {c.get('date')})")
        print(f"  config: {c.get('config')}")
        print(f"  encoder={c.get('encoder')} ce={c.get('ce_model')} kaynak={c.get('source')}")
    else:
        print("Henuz sampiyon kaydi yok. Bir eval kosup champion.update() cagir.")
