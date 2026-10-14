"""Son kaynak, ham olcum ve DB durumunu ayri okumalarla dogrula."""
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import statistics

kok=Path(__file__).resolve().parent
cikti={}
for etiket in ('v2','v3','gpu1','gpu2'):
    r=json.loads((kok/f'sonuc-{etiket}.json').read_text())
    assert r['sonuclar_ve_sayaclar_esit']
    assert len(r['olcumler'])==504
    assert len(r['sorgu_sha256'])==12
    for dal,h in r['kaynak_hashleri'].items():
        assert hashlib.sha256((kok/dal/'mnemonics/store.py').read_bytes()).hexdigest()==h
    assert hashlib.sha256((kok/'protokol.json').read_bytes()).hexdigest()==r['protokol_sha256']
    durumlar=[]
    for dal in ('once','sonra'):
        db=sqlite3.connect(f'file:{kok}/{etiket}-{dal}/memories.db?mode=ro',uri=True)
        assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        # Sadece duvar saati last_accessed farki disarida birakilir.
        durumlar.append(db.execute('SELECT id,ns,text,summary,meta,created,tier,access_count FROM memories ORDER BY id').fetchall())
        db.close()
    assert durumlar[0]==durumlar[1] and len(durumlar[0])==11580
    ozet={'encoder':r['encoder_device'],'veritabani_esit':True,
          'benzersiz_sorgu':len(set(r['sorgu_sha256'])),'yollar':{},'wal':r['wal']}
    for yol in ('vektor','hibrit','tam_retrieve'):
        c={}
        for x in r['olcumler']:
            if x['yol']==yol:
                anahtar=(x['sorgu'],x['tekrar'])
                assert x['dal'] not in c.setdefault(anahtar,{})
                c[anahtar][x['dal']]=x['ns']
        assert set(c)=={(q,t) for q in range(12) for t in range(7)}
        assert all(set(a)=={'once','sonra'} for a in c.values())
        farklar=[statistics.median((a['once']-a['sonra'])/a['once']
                                  for (q,t),a in c.items() if q==i) for i in range(12)]
        rng=random.Random(991)
        ornek=sorted(statistics.median(rng.choices(farklar,k=12)) for _ in range(10000))
        ozet['yollar'][yol]={**r['ozet'][yol],
                            'sorgu_bazli_medyan_azalma':statistics.median(farklar),
                            'sorgu_bootstrap_95':[ornek[250],ornek[9750]]}
    cikti[etiket]=ozet
orijinal=json.loads((kok/'kaynak.json').read_text())
assert hashlib.sha256((kok/'veri/memories.db').read_bytes()).hexdigest()==orijinal['veri_sha256']
assert hashlib.sha256((kok/'veri/index_sessions.bin').read_bytes()).hexdigest()==orijinal['index_sha256']
with (kok/'onay.json').open('x') as f:json.dump(cikti,f,indent=2)
for ad,r in cikti.items():
    print(ad,r['encoder'],'DB_EQUAL',r['benzersiz_sorgu'],'benzersiz sorgu')
    for yol,x in r['yollar'].items():
        print(yol,round(x['once'],3),round(x['sonra'],3),
              'azalma%',round(x['sure_azalmasi']*100,2),
              'sorgu_CI%',[round(v*100,2) for v in x['sorgu_bootstrap_95']])
print('ONAY_PASS: 2016 olcum; 4 veritabani cifti esit; kaynak hashleri esit')
