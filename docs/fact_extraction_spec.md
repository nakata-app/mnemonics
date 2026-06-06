# Fact-Extraction Katmanı, Spec v0.1 (2026-06-05)

## Hedef (ölçülebilir)
LoCoMo end-to-end QA accuracy **0.742 → ≥0.85** (DeepSeek judge, kategori 1-4,
aynı protokol). İkincil: retrieval R@5 0.772 → ≥0.85.

## Neden
- LoCoMo'da 89-92 bandındaki sistemlerin (Mem0, Hindsight) ortak özelliği:
  ham konuşma yerine **LLM ile damıtılmış atomik fact'ler** üzerinden retrieval.
  Bizim ham-turn retrieval, soru dili ile konuşma dili arasındaki mesafede
  kayıp veriyor (ölçüldü: turn-seviyesi R@1 0.549).
- Pazar trendi aynı yönde (episodic+semantic katmanlama); bu katman aynı
  zamanda satılabilir ürün özelliği.

## Kimlik ilkesi (PAZARLIKSIZ)
mnemonics = "verified AI memory". Bu katman kimliği SULANDIRMAZ, güçlendirir:
- **Opsiyonel**: çekirdek LLM'siz kalır. `ingest(..., extract_facts=True)`
  açık istekle çalışır; default kapalı.
- **Provenance zorunlu**: her fact, kaynak turn-chunk id'lerine bağlanır.
  Fact'ten kanıta tek atlama, Mem0'ın yapamadığı şey: *doğrulanabilir fact*.
- Extraction LLM'i konfigüre edilebilir (DeepSeek default; lokal model olabilir).

## Tasarım
```
ingest(texts) ──► raw chunks (mevcut yol, değişmez)
       └─(opsiyonel)─► extractor LLM ──► atomic facts
                        her fact: {text, source_chunk_ids, ts, kind}
                        aynı store'a typed row olarak (meta.kind="fact")
retrieve(query) ──► birleşik aday havuzu (fact + raw), tek rerank
                    fact hit'i cevap prompt'una kaynağıyla gider
```

## Dilimler
- **S1**: `mnemonics/extract.py`, extractor (batch, JSON-schema çıktı,
  maliyet sayacı, retry) + store'a typed-meta desteği. Birim testli.
- **S2**: LoCoMo koşusu extraction'lı ingest ile (Kaggle + DeepSeek) →
  judge → 0.742 baseline'a karşı A/B. KARAR NOKTASI: ≥+5pp değilse dur.
- **S3**: retrieval karışım ayarı (fact/raw oranı, dedup) + ikinci A/B.
- **S4**: docs + örnek + maliyet tablosu. (LME'ye dokunulmaz, orada tavandayız.)

## Maliyet/etki
- Geliştirme: S1-S2 ≈ 1 gün; S3-S4 ≈ yarım gün. [TAHMİN]
- API: LoCoMo full ingest ~190 session × extraction ≈ $0.3-0.8/koşu. [TAHMİN]
- Dokunulan dosyalar: extract.py (yeni), store.py (meta tip, küçük),
  retrieve.py (karışım, küçük), locomo köprüsü.

## Geri alma
Feature-flag (default off). Çekirdek yollar değişmediği için flag kapalıyken
davranış bugünkü ile bit-bit aynı; revert = flag'i kaldır + extract.py sil.

## Premortem (6 ay sonra battıysa neden?)
1. **Extraction halüsinasyonu**, fact kaynakta olmayan şey söylüyor.
   Önlem: provenance + (S3'te) fact↔kaynak entailment kontrolü opsiyonu.
   Tespit: örneklemeli fact-audit script'i. Sahip: extractor prompt'u.
2. **Maliyet sürünmesi**, büyük korpusta ingest faturası. Önlem: batch +
   token sayacı + "fact'siz devam" fallback. Tespit: maliyet/1k-turn metriği.
3. **Kimlik erozyonu**, "LLM'siz" iddiası zayıflar. Önlem: default-off +
   README'de net ayrım ("core: LLM-free; fact layer: optional, verified").
4. **LoCoMo'ya overfit**, başka veri setinde fayda yok. Önlem: S2 kararını
   LifeBench mini-örneklemiyle çapraz kontrol (ileride).
