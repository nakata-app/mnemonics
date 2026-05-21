# ⛔ DEPRECATED, STAGE-2 ÖLDÜ (2026-05-21)

**BU BRIEF'İ UYGULAMA. ATAKAN REVERT ETTİ.**

Stage-2 session-level indexing denendi (commits 868cac5 / f8cffd2 / 3b58c32) ve LME 500q'da **R@1 0.846 → 0.804 regress** verdi (single-session-preference -17pp, NETO ZERO lift). 3 commit reverted: 2f0a559, d9d157c, f15a495.

**Yeni yön:** CE rerank model upgrade (bge-reranker-v2-m3) → HyDE-only → per-q-type routing. Detay: konuşma transcript'i, peer koordinasyon notu.

**Eğer bu dosyayı yeni session'da okuyup başlamak üzereysen: DURDUR.** Konuşmaya dön, mevcut state'i sor.

---

# ~~Görev: Session-Level Indexing (Option B)~~ [ARTIK GEÇERSİZ]

**Hedef:** LME 500q'da R@1 > 0.90, R@10 > 0.95, MemPalace R@1=0.920/R@10=1.000 eşitle ya da geç.

---

## Neden bu değişiklik gerekli

Mevcut durum: mnemonics metinleri 200 kelimelik parçalara böler ve her parçayı ayrı ayrı indexler.
LME sorusu: "bu bilgi hangi SESSION'da konuşuldu?" diye sorar.
Sorun: parça-seviyesi arama session sınırlarını bilmiyor. Doğru session'ın parçaları global top-k'ya giremeyebilir.

MemPalace her session'ı tek bir belge olarak indexliyor. Biz bunu chunk indexinin yanına ekleyeceğiz (iki aşamalı arama).

Kanıt: augment_assistant_facts + AdaptMem FT encoder ikisi de sıfır lift verdi (Colab'da doğrulandı, 2026-05-20). Root cause mimarinin kendisi.

---

## Mevcut sonuçlar (referans)

```
mnemonics_rerank  n=500  R@1=0.846  R@5=0.898  R@10=0.898
MemPalace         n=500  R@1=0.920  R@5=0.960  R@10=1.000
```

---

## Değiştirilecek 3 dosya

### 1. `mnemonics/store.py`

`_SCHEMA` string'ine `documents` tablosunu ekle (mevcut `memories` tablosunun hemen altına):

```sql
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ns          TEXT NOT NULL DEFAULT 'default',
    text        TEXT NOT NULL,
    source_idx  INTEGER NOT NULL DEFAULT 0,
    meta        TEXT NOT NULL DEFAULT '{}',
    created     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_docs_ns ON documents(ns, source_idx);
```

`Store.__init__()` içine ekle (mevcut `self._index: dict` satırının yanına):
```python
self._doc_index: dict[str, hnswlib.Index] = {}
self._doc_index_mtime: dict[str, float] = {}
```

Yeni method `_doc_index_for(self, ns)`, `_index_for` ile aynı ama `index_{ns}_docs.bin` dosyasını kullanır.

Yeni method `add_doc(self, texts, vectors, ns, source_idxs, meta)`:
- `documents` tablosuna INSERT
- `_doc_index_for(ns)` HNSW'ye ekle
- `index_{ns}_docs.bin` olarak kaydet
- File lock pattern'i `add()` ile aynı

Yeni method `search_docs(self, vector, ns, top_k) -> list[int]`:
- `_doc_index_for(ns)` knn_query
- SQLite'ten `source_idx` değerlerini çek
- `list[int]` (source_idx listesi) döndür, score değil

`_ns_file_lock` ve `_reload_if_stale` doc indeksi için de çalışmalı, `_doc_index_for` içinde `_doc_index_mtime` kullan.

### 2. `mnemonics/ingest.py`

`ingest()` fonksiyonunun sonuna, `store.add()` çağrısından SONRA ekle:

```python
# Document-level embeddings: her input text için ilk 400 kelime → tek bir doc vektörü
doc_texts: list[str] = []
doc_source_idxs: list[int] = []
doc_metas: list[dict] = []
for i, text in enumerate(texts):
    words = text.split()
    doc_text = " ".join(words[:400])
    m = (meta[i] if meta else {}) | {"source_idx": i}
    doc_texts.append(doc_text)
    doc_source_idxs.append(i)
    doc_metas.append(m)

if doc_texts:
    doc_vecs = enc.encode(doc_texts, batch_size=64, show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True)
    store.add_doc(doc_texts, doc_vecs, ns=ns, source_idxs=doc_source_idxs, meta=doc_metas)
```

### 3. `mnemonics/retrieve.py`

`retrieve()` fonksiyonuna yeni parametre: `use_doc_filter: bool = False`

Retrieve fonksiyonunun başında, `enc.encode` satırından sonra:

```python
# Stage 1: Belge-seviyesi arama (hangi session'lar alakalı?)
candidate_source_idxs: set[int] | None = None
if use_doc_filter:
    candidate_source_idxs = set(store.search_docs(qvec, ns=ns, top_k=candidate_k))
```

`store.search()` ve `store.search_bm25()` çağrılarından dönen sonuçları filtrele:
```python
if candidate_source_idxs is not None:
    results = [r for r in results if r["meta"].get("source_idx") in candidate_source_idxs]
```

Bu filtreyi RRF fusion'dan ÖNCE hem vec_results hem bm25_results'a uygula.

---

## Eval komutu (değişiklik sonrası)

```bash
cd /Users/macmini/Projects/mnemonics
PYTHONUNBUFFERED=1 .venv/bin/python -u benchmarks/longmemeval_eval.py \
  --n 50 --mode rerank \
  --augment-preferences --candidate-k 50 --seed 42 \
  --out /tmp/lme_stage2_50q.json
```

`longmemeval_eval.py` içindeki `evaluate_mnemonics()` çağrılarına `use_doc_filter=True` parametresini ekle.

Smoke test (5 soru, hızlı kontrol): `--n 5 --mode no_rerank`

---

## Başarı kriterleri

1. Tüm mevcut testler geçer: `pytest tests/ -x -q`
2. `store.add_doc()` ve `store.search_docs()` için en az 2 unit test
3. Smoke test (5q) hatasız çalışır
4. 50q eval: R@1 >= 0.88, R@10 >= 0.92 (mevcut 0.846/0.898'den yukarı)
5. 500q eval hedef: R@1 > 0.90, R@10 > 0.95

---

## Dikkat: API key güvenliği

`DEEPSEEK_API_KEY` değeri `sk-ebce...` ile başlar. Bu değer git'e yazılmaz.
`locomo_run.py` Kaggle'da `os.environ["DEEPSEEK_API_KEY"]` ile okur, ortam değişkeni olarak set edilmeli.
`.gitignore`'a `.env` ve `*_key.txt` eklenmiş mi kontrol et.

---

## Dosya yolları

- Eval harness: `benchmarks/longmemeval_eval.py`
- LME dataset: `/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json`
- Store: `mnemonics/store.py`
- Ingest: `mnemonics/ingest.py`
- Retrieve: `mnemonics/retrieve.py`
- Testler: `tests/`
- Venv: `.venv/bin/python`

---

## Neden bu yaklaşım

MemPalace'ın tam session-replace yaklaşımı yerine iki katmanlı: chunk indexi korunuyor (LoCoMo gibi fragment-query'lerde iyi çalışıyor), üstüne session-level filtre ekleniyor. Mevcut `source_idx` meta alanı zaten chunk'larda var, ekstra schema gerekmez, sadece doc index tablosu ekleniyor.

Premortem zayıf halka: `candidate_source_idxs` filtresi çok agresif olursa (yanlış session'ı dışarıda bırakırsa) R@k düşer. Çözüm: `candidate_k=50` ile geniş doc filtresi (50 session içinden filtrele, zaten tüm LME haystack'i bu kadar).
