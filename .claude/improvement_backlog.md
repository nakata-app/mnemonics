# mnemonics improvement backlog
# Otonom ajan bu dosyadan görev alır. Tamamlananlar [x] işaretlenir.
# Sıra önemli: yukarıdan aşağıya işle.

## [x] 1. store.delete() vector orphan fix (BUG) (2026-06-18)
**Hedef:** `store.delete(id)` şu an sadece SQLite'tan siliyor, hnswlib vector index'ini güncellemıyor.
Silinmiş ID'ler index'te orphan olarak kalıyor, retrieve sırasında hayalet sonuç dönüyor.

**Dosyalar:**
- `mnemonics/store.py`, `Store.delete()` metodu (~satır 415-419)
- `mnemonics/store.py`, `_index_for()` ve `_save_index()` metotları

**Yapılacak:**
1. `Store.delete(id)` metodunda SQL sildikten sonra:
   - O row'un `ns`'ini önceden sorgula (silerken kaybolacak)
   - İlgili ns'in hnswlib index'ini yeniden oluştur (rebuild): ns'teki kalan tüm row'ların embedding'lerini çekip index'i sıfırdan yaz
   - Ya da hnswlib'nin `mark_deleted(label)` API'sini kullan (soft-delete, daha hızlı)
2. `gc()` metodunu da aynı şekilde düzelt (toplu delete)
3. Test: ingest → retrieve (ID var) → delete → retrieve (ID yok) → index'te de yok

**Doğrulama kriteri:** `mnemonics retrieve` silinen ID'yi bir daha döndürmüyor.

---

## [x] 2. mnemonics forget komutu (OPERABILITY) (2026-06-18)
**Hedef:** Raw SQL kullanmadan namespace temizliği yapabilmek.
Bugün `sessions` ns'ini temizlemek için doğrudan sqlite3 kullandık, bu CLI'dan yapılabilmeli.

**Dosyalar:**
- `mnemonics/cli.py`, yeni subparser ekle
- `mnemonics/store.py`, `Store.forget(ns, before_date=None, tier=None)` metodu

**Yapılacak:**
```
mnemonics forget --ns sessions                          # tüm ns'i sil
mnemonics forget --ns sessions --before 2026-01-01     # tarihten eskiyi sil
mnemonics forget --ns sessions --tier 1                # sadece tier=1'i sil
mnemonics forget --ns sessions --dry-run               # ne silineceğini göster, silme
```
Store.forget() SQL DELETE + hnswlib index rebuild içermeli (Task 1 fix'i kullan).

**Doğrulama kriteri:** `mnemonics forget --ns test --dry-run` çıktı üretir, `--apply` ile siler.

---

## [x] 3. mnemonics doctor komutu (OPERABILITY) (2026-06-18)
**Hedef:** DB sağlık kontrolü: bozuk index, orphan .bin dosyası, sql/vector count uyumsuzluğu.

**Dosyalar:**
- `mnemonics/cli.py`, yeni subparser
- `mnemonics/store.py`, `Store.health_check()` metodu

**Kontroller:**
1. SQLite `PRAGMA integrity_check`
2. Her ns için: SQL count vs hnswlib index `element_count` karşılaştır
3. `.bin` dosyası olan ama DB'de hiç row'u olmayan namespace'ler (orphan index)
4. DB'de row'u olan ama `.bin` dosyası olmayan namespace'ler (eksik index)
5. WAL dosyası boyutu (büyükse checkpoint öner)

**Doğrulama kriteri:** `mnemonics doctor` çalışıp OK veya sorunları raporluyor.

---

## [x] 4. gc tier-1 desteği (OPERABILITY) (2026-06-18)
**Hedef:** Şu an `mnemonics gc` sadece tier-2 (ambient) siliyor. `sessions` ns gibi tier-1 çöpü için
raw SQL kullanmak zorunda kalıyoruz.

**Dosyalar:**
- `mnemonics/store.py`, `gc_candidates()` ve `gc()` metotları
- `mnemonics/cli.py`, `gc` subparser'a `--tier` flag ekle

**Yapılacak:**
```
mnemonics gc --tier 1 --ns sessions --age-days 60 --apply
```
gc_candidates'e `tier` parametresi ekle (None = sadece tier-2, 1 = tier-1 dahil).

**Doğrulama kriteri:** `mnemonics gc --tier 1 --ns sessions --age-days 60` doğru aday sayısı gösteriyor.

---

## [x] 5. bge-reranker-v2-m3 cross-encoder upgrade (QUALITY) (2026-06-18)
**Hedef:** Mevcut cross-encoder'ı `BAAI/bge-reranker-v2-m3` ile değiştir.
LongMemEval-S'de single-session-preference darboğazı (R@1=0.70 → hedef 0.85+).

**Dosyalar:**
- `mnemonics/retrieve.py`, reranker yükleme kısmı
- `mnemonics/store.py`, `MNEMONICS_RERANK_MODEL` env var kullanımı

**Yapılacak:**
1. retrieve.py'de reranker model path/id env var'dan okunuyor mu kontrol et
2. Default model'i `BAAI/bge-reranker-v2-m3` olarak güncelle (eski: cross-encoder/ms-marco-MiniLM-L-6-v2 ya da adaptmem)
3. README'de `MNEMONICS_RERANK_MODEL` env var'ını belgele
4. LongMemEval-S üzerinde eval çalıştır: `mnemonics eval ...` ya da adaptmem eval script

**Doğrulama kriteri:** `mnemonics retrieve --rerank` bge-reranker-v2-m3 kullanıyor, eval R@1 ≥ 0.964.

---

## Bekleyen (onay gerekli)

- `zeus_premortem` (25 satır) → `proj:zeus` (19 satır) ile birleştirme? Atakan kararı lazım, ayrı tutmak vs merge.
- `test` (5), `testdecay` (1), vb. küçük namespace'ler silinsin mi? Destructive, onay lazım.

## Tamamlananlar

### [x] 3. mnemonics doctor (2026-06-18)
Store.health_check() + CLI `doctor [--json]`. DB integrity, WAL, per-ns sql/idx fark, orphan .bin.
Canlı: sessions 10935 orphan, proj:AdaptMem 24 orphan, proj:Database 1 missing vector tespiti.
231/231 geçti.

### [x] 2. mnemonics forget komutu (2026-06-18)
Store.forget() + forget_candidates() + CLI `forget --ns X [--before DATE] [--tier N] [--apply]`.
7 yeni test. 228/228 geçti.

### [x] 1. store.delete() vector orphan fix (2026-06-18)
store.delete() + gc() → mark_deleted(id) + save_index. search() RuntimeError guard eklendi.
3 yeni test: test_delete_removes_from_vector_index, test_gc_removes_from_vector_index.
221/221 test geçti. Commit: fix: remove deleted entries from hnswlib vector index
