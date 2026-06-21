"""Champion eval — full production config, GPU. Run via `krun`.

Birlestirir: kaggle_eval1'in turn+temporal+augment config'i + kaggle_v2m3'un
guclu bge-reranker-v2-m3 CE'si. PEER_COORDINATION production config'i
(--chunk-mode turn --temporal-aware --candidate-k 50 --mode rerank --seed 42)
uzerine en guclu CE. Hedef: bilinen sampiyon R@1'i tek, tekrar edilebilir,
KAYITLI bir kosuda dogrulamak.

Olmeyecek sekilde: her dis adim retry'li, dataset boyut+JSON dogrulamali
(fail-fast), her asama (smoke->100->500) sonucu HEMEN diske yazilir. 500q
coker/timeout olursa 100q korunur. Sonuc /kaggle/working/results altinda,
krun geri ceker.

Kullanim:
  krun benchmarks/kaggle_champion.py --dataset atakanakbaba/mnemonics-lme --acc NvidiaTeslaT4
"""
import os, sys, json, time, subprocess

WORK = '/kaggle/working'
REPO = f'{WORK}/mnemonics'
RESULTS = f'{WORK}/results'
os.makedirs(RESULTS, exist_ok=True)

KAGGLE_DS = '/kaggle/input/mnemonics-lme/longmemeval_s_cleaned.json'
DATA = f'{WORK}/longmemeval_s.json'
URL = 'https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s'
MIN_BYTES = 100_000_000
CE_MODEL = 'BAAI/bge-reranker-v2-m3'


def retry(fn, what, tries=3, wait=8):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            print(f'[retry {i}/{tries}] {what} basarisiz: {e}', flush=True)
            if i < tries:
                time.sleep(wait * i)
            else:
                raise


# 1) Repo
def clone():
    subprocess.run(['rm', '-rf', REPO], check=False)
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/nakata-app/mnemonics.git', REPO], check=True)
retry(clone, 'git clone')
subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-3'])

# 2) Bagimliliklar
retry(lambda: subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO,
     'sentence-transformers', 'numpy', 'adaptmem'], check=True),
    'pip install')
print('Install OK', flush=True)

# 3) Dataset — fail-fast
def valid(p):
    return os.path.exists(p) and os.path.getsize(p) > MIN_BYTES

if valid(KAGGLE_DS):
    DATA = KAGGLE_DS
    print(f'Kaggle dataset: {DATA} ({os.path.getsize(DATA)/1e6:.1f} MB)', flush=True)
elif valid(DATA):
    print(f'Cache var: {DATA} ({os.path.getsize(DATA)/1e6:.1f} MB)', flush=True)
else:
    def dl():
        subprocess.run(['wget', '-q', '-O', DATA, URL], check=True)
        if os.path.getsize(DATA) < MIN_BYTES:
            raise RuntimeError(f'inen dosya cok kucuk: {os.path.getsize(DATA)} byte')
    retry(dl, 'dataset indirme', tries=3, wait=10)
    print(f'Indi: {os.path.getsize(DATA)/1e6:.1f} MB', flush=True)

with open(DATA) as f:
    nq = len(json.load(f))
print(f'Dataset OK: {nq} soru', flush=True)

# 4) Env — guclu CE
env = os.environ.copy()
env['LME_DATA'] = DATA
env['PYTHONUNBUFFERED'] = '1'
env['MNEMONICS_RERANK_MODEL'] = CE_MODEL

EVAL = 'benchmarks/longmemeval_eval.py'
# PEER_COORDINATION production config + augment-preferences
COMMON = ['--mode', 'rerank', '--chunk-mode', 'turn', '--temporal-aware',
          '--augment-preferences', '--candidate-k', '50', '--seed', '42']


def stage(n, tag):
    out = f'{RESULTS}/lme{n}_champion.json'
    perq = f'{RESULTS}/lme{n}_champion_perq.json'
    print(f'\n=== {tag} ({n}q) — turn+temporal+augment + {CE_MODEL} ===', flush=True)
    r = subprocess.run([sys.executable, '-u', EVAL, '--n', str(n), *COMMON,
                        '--out', out, '--per-q-out', perq], cwd=REPO, env=env)
    if r.returncode != 0 or not os.path.exists(out):
        print(f'[WARN] {tag} returncode={r.returncode} — sonuc yok', flush=True)
        return None
    res = json.load(open(out))['mnemonics_rerank']
    print(f'{tag}: R@1={res["R@1"]:.3f} R@5={res["R@5"]:.3f} R@10={res["R@10"]:.3f}', flush=True)
    return res


# 5) Smoke — bozuksa ANINDA dur
if stage(5, 'SMOKE') is None:
    print('SMOKE FAILED — uzun kosulardan once durduruldu.', flush=True)
    sys.exit(1)
print('SMOKE OK', flush=True)

# 6) 100q
r100 = stage(100, '100q')

# 7) 500q — coker/timeout olursa 100q korunur
try:
    r500 = stage(500, '500q')
except Exception as e:
    print(f'[500q coktu] {e} — 100q sonucu korundu', flush=True)
    r500 = None

# 8) Sampiyon ozeti (krun geri ceker, biz local'de kalici kaydederiz)
champion = {
    'config': ' '.join(COMMON), 'ce_model': CE_MODEL, 'encoder': 'all-MiniLM-L6-v2',
    'r100': r100, 'r500': r500,
}
json.dump(champion, open(f'{RESULTS}/champion_summary.json', 'w'), indent=2)
print('\n=== CHAMPION OZET ===', flush=True)
if r100:
    print(f'100q: R@1={r100["R@1"]:.3f} R@5={r100["R@5"]:.3f} R@10={r100["R@10"]:.3f}', flush=True)
if r500:
    print(f'500q: R@1={r500["R@1"]:.3f} R@5={r500["R@5"]:.3f} R@10={r500["R@10"]:.3f}', flush=True)
    print('\nBy type:', flush=True)
    for qt in sorted(r500['by_type']):
        b = r500['by_type'][qt]
        print(f'  {qt:28} n={b["n"]:3}  R@1={b["R@1"]:.3f}', flush=True)
print('\nPEER_COORDINATION baseline: R@1=0.954 (chunk-mode turn + rerank)', flush=True)
