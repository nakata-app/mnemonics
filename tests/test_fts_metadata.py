import importlib

import numpy as np
import pytest

from mnemonics.store import Store


@pytest.fixture
def depo(tmp_path):
    s = Store(tmp_path, dim=8)
    s.add(
        ["ozgunbir metin", "ikincibir metin"],
        np.ones((2, 8), dtype="float32"),
        summaries=[None, "ozetbir"],
    )
    yield s
    s._db.close()


def test_sayac_metadata_yazmasi_indeksi_yenilemez(depo):
    for atama in (
        "access_count=access_count+1",
        "last_accessed=datetime('now')",
        "tier=2",
        "meta='{}'",
        "text=text",
        "summary=summary",
    ):
        once = depo._db.total_changes
        depo._db.execute(f"UPDATE memories SET {atama} WHERE id=1")
        depo._db.commit()
        assert depo._db.total_changes - once == 1, atama
    assert depo.search_bm25("ozgunbir")[0]["id"] == 1


def test_metin_ozet_ve_id_degisikligi_indekse_yansir(depo):
    depo._db.execute("UPDATE memories SET text='yenimetin',summary='yeniozet' WHERE id=1")
    depo._db.commit()
    assert not depo.search_bm25("ozgunbir")
    assert depo.search_bm25("yenimetin")[0]["id"] == 1
    assert depo.search_bm25("yeniozet")[0]["id"] == 1
    depo._db.execute("UPDATE memories SET summary=NULL,id=19 WHERE id=1")
    depo._db.commit()
    assert not depo.search_bm25("yeniozet")
    assert depo.search_bm25("yenimetin")[0]["id"] == 19
    depo._db.execute("DELETE FROM memories WHERE id=19")
    depo._db.commit()
    assert not depo.search_bm25("yenimetin")
    depo._db.execute("INSERT INTO memories_fts(memories_fts,rank) VALUES('integrity-check',1)")


def test_eski_tetikleyici_bir_defa_guncellenir(depo):
    depo._db.execute("DROP TRIGGER memories_au")
    depo._db.executescript("""
      CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts,rowid,text,summary) VALUES('delete',old.id,old.text,old.summary);
        INSERT INTO memories_fts(rowid,text,summary) VALUES(new.id,new.text,new.summary);
      END;
    """)
    ikinci = Store(depo.root, dim=8)
    once = ikinci._db.total_changes
    ikinci._db.execute("UPDATE memories SET access_count=access_count+1 WHERE id=1")
    ikinci._db.commit()
    assert ikinci._db.total_changes - once == 1
    surum = ikinci._db.execute("PRAGMA schema_version").fetchone()[0]
    ucuncu = Store(depo.root, dim=8)
    assert ucuncu._db.execute("PRAGMA schema_version").fetchone()[0] == surum
    assert ucuncu.search_bm25("ozgunbir")[0]["id"] == 1
    ikinci._db.close()
    ucuncu._db.close()


def test_vektor_arama_sayaclari_korunur(depo):
    sonuc = depo.search(np.ones(8, dtype="float32"), top_k=2)
    assert len(sonuc) == 2
    assert depo._db.execute("SELECT sum(access_count) FROM memories").fetchone()[0] == 2


@pytest.mark.parametrize("alias", ["rowid", "_rowid_", "oid"])
def test_kimlik_takma_adlari_indeksi_korur(depo, alias):
    depo._db.execute(f"UPDATE memories SET {alias}=19 WHERE id=1")
    depo._db.commit()
    assert depo.search_bm25("ozgunbir")[0]["id"] == 19
    depo._db.execute("INSERT INTO memories_fts(memories_fts,rank) VALUES('integrity-check',1)")


def test_basarisiz_migrasyon_eski_tetikleyiciyi_korur(depo, monkeypatch):
    mod = importlib.import_module("mnemonics.store")
    eski = depo._db.execute("SELECT sql FROM sqlite_master WHERE name='memories_au'").fetchone()[0]
    monkeypatch.setattr(mod, "_FTS_UPDATE_TRIGGER", "CREATE TRIGGER bozuk INVALID SQL")
    with pytest.raises(Exception):
        depo._migrate_fts()
    assert (
        depo._db.execute("SELECT sql FROM sqlite_master WHERE name='memories_au'").fetchone()[0]
        == eski
    )
