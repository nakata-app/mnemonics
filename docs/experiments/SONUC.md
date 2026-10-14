# Mnemonics: her aramada yapılan gereksiz indeks yazmasını kaldırma

2026-09-13. **Gerçek proje verisinin kopyasında doğrulanmış fayda var.**
Apple M2'nin GPU yolu kullanıldığında sıcak `retrieve()` çağrısının medyan
süresi iki sorgu grubunda %3,66 ve %3,99 azaldı. Vektör araması yaklaşık
%44–45 hızlandı. Ayrı yazma ölçümünde SQLite işlem günlüğü (WAL) büyümesi
%42,53 ve %46,58 azaldı. Arama sonuçları ve erişim sayaçları korundu.

Bu geniş bir bilimsel keşif değildir. Mnemonics'in her normal vektör
aramasında yaptığı gereksiz bir işi kaldıran, gerçek veride ölçülmüş
mühendislik düzeltmesidir. Doğrulama sonrasında Atakan'ın talimatıyla
canlıya uygulandı; dağıtım kanıtı `canli-sonuc.json` içindedir.

## Gerçek problem

`Store.search()` döndürdüğü adayların `access_count` ve `last_accessed`
alanlarını güncelliyor. Canlı veritabanındaki `memories_au AFTER UPDATE`
tetikleyicisi ise güncellenen alanlara bakmadan metni FTS5 arama indeksinden
siliyor ve yeniden ekliyor. Metin değişmediği hâlde indeksleme çalışıyor.

Canlı veritabanı yalnızca okunarak bu tetikleyicinin varlığı doğrulandı.
Küçük bir regresyon testinde bir sayaç artışı, SQLite `total_changes`
sayacında 1 yerine 9 satır değişikliği oluşturdu. Bu sayı SQL sorgu
sayısı değil; tetikleyici ve FTS yan etkileri dahil değişiklik sayısıdır.

## Düzeltme

Yalnızca `old.id`, `old.text` veya `old.summary` yeni değerinden farklıysa
FTS güncelleniyor. `IS NOT`, NULL özetlerin de doğru karşılaştırılmasını
sağlıyor. Eski veritabanının tetikleyicisi açılışta bir defa, savepoint
içinde güncelleniyor; indeksin içeriği baştan oluşturulmuyor. Zaten güncel
tetikleyici tekrar değiştirilmediği için her açılışta şema yazması yok.

Kimlik `id` yerine `rowid`, `_rowid_` veya `oid` adıyla değiştirildiğinde
de indeks tutarlı kalır. İlk adayda kullanılan `UPDATE OF` sınırının bu
durumu kaçırdığı ek testlerle bulundu ve kaldırıldı. İlk adayın ölçümü
`sonuc.json` içinde korunuyor ama nihai bulguya dahil edilmiyor.

Değişiklik `sonra/mnemonics/store.py` içinde. `degisiklik.patch`, kaynak
değişikliğiyle birlikte yeni regresyon testlerini de içerir.

## Kaynak ve veri

- Ana depo: `/Users/macmini/Projects/mnemonics`.
- Git HEAD: `4142db13f4cbcbe44479ed5d6605e543de63dbbb`.
- Başlangıç salt HEAD değildir: mevcut kaydedilmemiş değişiklikleri içeren
  çalışma ağacının anlık kopyasıdır. Her kaynak dosyasının hash'i
  `kaynak.json` içindedir. Başkalarının değişikliklerine dokunulmadı.
- Canlı DB: 11.580 anı; sessions alanında 10.950 anı.
- SQLite backup API ve paylaşılan namespace kilidiyle yerel kopya alındı.
  Aramalardaki sayaç güncellemeleri yalnızca kopyalarda yapıldı.
- Kaynak DB ve HNSW görüntüsünün hash'leri deney sonunda tekrar doğrulandı.
- Her grupta, gerçek anılardan sabit tohumla seçilen 12 metnin ilk yedi
  kelimesi sorgu olarak kullanıldı. İki ayrı tohum: 2026091307/2026091308.
  **Bunlar canlı sorgu trafiğinin kaydı değildir.**
- Anı metinleri/sorgular rapora konmadı; ham ölçümlerde sorgu hash'leri var.
  Model ve veriler yerelde kaldı; model indirme kapalıydı.

## Ölçüm düzeni

`protokol.json` ölçüm öncesinde kaydedildi. Aynı veri görüntüsünden açılan
iki Store örneği, her sorguyu yedi defa işledi. A/B sırası ve sorgu sırası
rastgele değiştirildi. Yavaş tekrarlar çıkarılmadı. Son sürüm için:

- 2 sorgu grubu × 2 çalışma ortamı × 3 arama yolu × 12 sorgu × 7 tekrar
  × 2 sürüm = **2.016 ham çağrı ölçümü**.
- Vektör yolu: gerçek HNSW araması, SQL sonucu alma ve erişim yazması.
- Hibrit yol: `Store.hybrid_search()`, vektör + BM25 + birleştirme.
- Tam yol: mevcut yerel encoder dahil gerçek `retrieve()` Python API'si;
  varsayılan `candidate_k=50`, `top_k=5`, rerank kapalı.
- Model ve indeks ilk yüklenmesi ısınmada kaldı. Sonuçlar uzun yaşayan
  süreçteki sıcak çağrılar içindir. MCP taşıması, ağ ve sonraki dil modeli
  cevabı bu ölçüme dahil değildir.

Sandbox içinde PyTorch CPU seçti. Normal yerel ortamda MPS GPU erişimi
doğrulandı ve aynı deney orada tekrarlandı. CPU sonucu GPU sonucu gibi
sunulmadı; iki ortamın ham kayıtları da korundu. Her karşılaştırmada aynı
encoder, aynı HNSW görüntüsü ve aynı arama parametreleri kullanıldı.

## Son sürümün sonuçları

Süreler milisaniye; her hücre 84 çağrının medyanı.

| Ortam / sorgu grubu | Yol | Önce | Sonra | Süre azalması |
|---|---|---:|---:|---:|
| GPU / 1 | Vektör | 2,858 | 1,571 | %45,03 |
| GPU / 1 | Hibrit | 16,165 | 15,198 | %5,98 |
| GPU / 1 | Tam retrieve | 46,638 | 44,933 | %3,66 |
| GPU / 2 | Vektör | 3,013 | 1,677 | %44,35 |
| GPU / 2 | Hibrit | 15,347 | 14,653 | %4,52 |
| GPU / 2 | Tam retrieve | 45,405 | 43,592 | %3,99 |
| CPU / 1 | Tam retrieve | 98,775 | 97,206 | %1,59 |
| CPU / 2 | Tam retrieve | 100,292 | 98,188 | %2,10 |

Yedi tekrarı bağımsız sorgular gibi saymamak için belirsizlik ayrıca
sorgu bazında hesaplandı: önce her sorgunun eşleşmiş süre azalmalarının
medyanı alındı, sonra 12 sorgu üzerinde 10.000 bootstrap örneklemi yapıldı.
GPU tam çağrı için bu ayrı istatistiğin %95 aralıkları grup 1'de
%3,04–4,22, grup 2'de %3,25–5,10. Bunlar yukarıdaki medyanların oranıyla
birebir aynı istatistik değildir. Yalnızca bu sorgu grupları ve makine
için belirsizlik tahminidir; tüm üretim trafiği için garanti değildir.

## Yazma işi

Her dalda ayrı 12 vektör sorgusu çalıştırıldı, her sorgu 50 adayın
erişim sayacını güncelledi. Önce WAL checkpoint yapıldı; bu ayrı aşamada
autocheckpoint kapatılarak günlük büyümesi ölçüldü. Süre ölçümlerinde
uygulamanın varsayılan checkpoint davranışı korundu.

| Sorgu grubu | WAL önce (bayt) | WAL sonra (bayt) | Azalma | total_changes önce → sonra |
|---|---:|---:|---:|---:|
| 1 | 3.806.912 | 2.187.752 | %42,53 | 3.362 → 600 |
| 2 | 4.280.712 | 2.286.632 | %46,58 | 4.114 → 600 |

SQLite değişiklik sayısı %82,15–85,42 azaldı. Kalan 600 değişiklik,
12 × 50 gerçek erişim sayacı güncellemesidir. WAL baytı, fiziksel SSD
yazması veya bütün sistemin disk trafiği değildir. Enerji, SSD ömrü ve
çoklu oturum kapasitesi ölçülmedi; bunlar için kazanç iddiası yapılmıyor.

## Doğruluk ve güvenli geçiş

- Sayaç/metadata/sabit metin güncellemeleri FTS'ye yazmıyor.
- Metin/özet/kimlik değişiklikleri, NULL özetler, INSERT/DELETE ve üç
  rowid takma adı arama indeksine doğru yansıyor.
- Eski tetikleyici bir defa değişiyor; sonraki açılış şema sürümünü artırmıyor.
- Hatalı migrasyon simülasyonunda eski tetikleyici savepoint ile korunuyor.
- Mevcut store/hybrid/peer testleri ve yeni regresyonlar: **496 geçti**.
- Her A/B çağrısında sonuç alanları, sıra, skorlar ve sayaçlar karşılaştırıldı.
  Yalnızca duvar saatine bağlı `last_accessed` metni karşılaştırma dışında.
- Ayrı analiz kodu dört DB çiftinin 11.580 kaydını yeniden okuyup
  `last_accessed` dışındaki tüm içerik/metadata/sayaç alanlarının eşitliğini
  doğruladı. Her DB'de quick_check; ölçümlerde FTS integrity-check geçti.
- Son sürümün kaynak hash'i dört nihai ham kayıtla eşleşti.
- `git apply --check` ana deponun mevcut durumu üzerinde geçti.

## Tekrar üretme ve uygulama durumu

Python ortamı: `/Users/macmini/.local/pipx/venvs/mnemonics/bin/python`.
`deney.py` çalıştırırken `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` ve
`PYTHONDONTWRITEBYTECODE=1` kullanıldı. Normal yerel macOS ortamında model
MPS GPU'yu seçer; bu sandbox içinde CPU seçti. Yeni koşu etiketi ve yeni
çıktı yolları kullanılmalı; mevcut kayıtların üzerine yazılmaz.

```sh
cd /Users/macmini/Projects/mnemonics-fts-deneyi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonra /Users/macmini/.local/pipx/venvs/mnemonics/bin/python -m pytest -p no:cacheprovider -q test_fts_metadata.py sonra/tests/test_store.py sonra/tests/test_hybrid.py sonra/tests/test_peer.py
git -C /Users/macmini/Projects/mnemonics apply --check "$PWD/degisiklik.patch"
```

Her yeni kaynak/çıktı dosyası için çalışma alanının svelka-lock kuralını
uygula. `hazirla.py` kaynakları ve veriyi ilk kez kopyalar; mevcut deney
üzerinde yeniden çalıştırılmamalı. `deney.py <yeni-etiket> <tohum>` yeni
veri kopyaları oluşturur. `analiz.py` saklanan nihai dört koşuyu bağımsız
okur; `onay.json` zaten varsa korumak için yazmayı reddeder.

İlk deney ana depoya dokunmadan tamamlandı. Atakan'ın sonraki açık
canlıya alma talimatıyla yalnızca test edilen yama ana depoya uygulandı;
mevcut diğer değişiklikler korundu ve 496 test tekrar geçti. Kaynak dosya
ile canlı DB'nin dağıtım öncesi yedekleri alındı. Ardından test edilen
migrasyon canlı DB'ye tek işlem içinde uygulandı. Bu sırada mevcut
11.581 kaydın tüm içerik ve sayaç değerleri korundu. Canlı sayaç denemesi
geri alındı; sadece 1 SQL değişikliği oluştuğu görüldü. FTS arama sonucu
ve bütünlük testi geçti. Ayrı salt okunur bağlantı yeni tetikleyiciyi
doğruladı. Mevcut süreçler DB tetikleyicisini kullandığından servis
yeniden başlatılmadı. Dağıtım sonrası hız yeniden ölçülmedi; yukarıdaki
yüzdeler kontrollü kopya deneyine aittir.

## Kanıtlar

- `kaynak.json`: kaynak ve veri görüntüsü kimlikleri.
- `protokol.json`: ölçüm öncesi koşullar.
- `sonuc-v2.json`, `sonuc-v3.json`: son sürümün CPU ölçümleri.
- `sonuc-gpu1.json`, `sonuc-gpu2.json`: son sürümün GPU ölçümleri.
- `onay.json`, `analiz.py`: bağımsız durum ve istatistik denetimi.
- `degisiklik.patch`: tek üretim dosyası ve yeni regresyon testlerinin yaması.

Ön durum: deney klasörü yoktu; regresyon testleri eski davranışta başarısızdı.
Son durum: testler geçti, dosyalar ayrı okumayla doğrulandı, 2.016 kayıt
ve dört eşit DB çifti mevcut. Dosya/ölçüm bütünlüğü: PASS. Gereksiz FTS
yazmasını kaldırma hedefi: PASS. Canlı dağıtım: yapılmadı.

SQLite davranışı için birincil kaynaklar:
[CREATE TRIGGER](https://www.sqlite.org/lang_createtrigger.html),
[FTS5 dış içerik tabloları](https://www.sqlite.org/fts5.html#external_content_tables).
