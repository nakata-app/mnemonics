"""Eval 1 — default MiniLM CE, turn mode, CPU. Kaggle (krun --no-gpu).

 Olmeyecek sekilde: her dis adim retry'li, dataset URL'i dogru + boyut +
JSON dogrulamali (bos/yarim inerse ANINDA durur, saatler sonra degil), her
eval asamasi sonucu hemen diske yazilir (smoke->100->500). 500q coker/timeout
olursa 100q sonucu korunur.
"""
import os, sys, json, time, subprocess

WORK = '/kaggle/working'
REPO = f'{WORK}/mnemonics'
RESULTS = f'{WORK}/results'
os.makedirs(RESULTS, exist_ok=True)

# Dogru HF dosyasi: repo lowercase'e tasinmis, dosya adi 'longmemeval_s'
# (uzantisiz). Eski '..._cleaned.json' / '..._s.json' -> 404 (onceki olum sebebi).
DATA = f'{WORK}/longmemeval_s.json'
KAGGLE_DS = '/kaggle/input/mnemonics-lme/longmemeval_s_cleaned.json'
URL = 'https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s'
MIN_BYTES = 100_000_000


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


# 3) Dataset — idempotent + dogru URL + boyut + JSON kontrolu (fail-fast)
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
        sz = os.path.getsize(DATA)
        if sz < MIN_BYTES:
            raise RuntimeError(f'inen dosya cok kucuk: {sz} byte (hata sayfasi mi indi?)')
    retry(dl, 'dataset indirme', tries=3, wait=10)
    print(f'Indi: {os.path.getsize(DATA)/1e6:.1f} MB', flush=True)

with open(DATA) as f:
    nq = len(json.load(f))
print(f'Dataset OK: {nq} soru', flush=True)

# 4) Env — default CE (MiniLM); rerank-model override YOK (Eval 1 baseline)
env = os.environ.copy()
env['LME_DATA'] = DATA
env['PYTHONUNBUFFERED'] = '1'

EVAL = 'benchmarks/longmemeval_eval.py'
COMMON = ['--mode', 'rerank', '--chunk-mode', 'turn', '--temporal-aware',
          '--augment-preferences', '--candidate-k', '50', '--seed', '42']


def stage(n, tag):
    out = f'{RESULTS}/lme{n}_eval1_minilm.json'
    perq = f'{RESULTS}/lme{n}_eval1_minilm_perq.json'
    print(f'\n=== {tag} ({n}q) — default MiniLM CE + turn ===', flush=True)
    r = subprocess.run([sys.executable, '-u', EVAL, '--n', str(n), *COMMON,
                        '--out', out, '--per-q-out', perq], cwd=REPO, env=env)
    if r.returncode != 0 or not os.path.exists(out):
        print(f'[WARN] {tag} returncode={r.returncode} — sonuc yok', flush=True)
        return None
    res = json.load(open(out))['mnemonics_rerank']
    print(f'{tag}: R@1={res["R@1"]:.3f} R@5={res["R@5"]:.3f} R@10={res["R@10"]:.3f}', flush=True)
    return res


# 5) Smoke — bozuksa saatlerce beklemeden ANINDA dur
if stage(5, 'SMOKE') is None:
    print('SMOKE FAILED — uzun kosulardan once durduruldu.', flush=True)
    sys.exit(1)
print('SMOKE OK', flush=True)

# 6) 100q (hemen kaydedildi)
r100 = stage(100, '100q')

# 7) 500q — coker/timeout olursa 100q korunur
try:
    r500 = stage(500, '500q')
except Exception as e:
    print(f'[500q coktu] {e} — 100q sonucu korundu', flush=True)
    r500 = None

# 8) Ozet
print('\n=== EVAL-1 (MiniLM CE) CPU OZET ===', flush=True)
if r100:
    print(f'100q: R@1={r100["R@1"]:.3f} R@5={r100["R@5"]:.3f} R@10={r100["R@10"]:.3f}', flush=True)
if r500:
    print(f'500q: R@1={r500["R@1"]:.3f} R@5={r500["R@5"]:.3f} R@10={r500["R@10"]:.3f}', flush=True)
print('Ref MiniLM eski: R@1=0.846 | Colab eval2 (bge-v2-m3) ile karsilastir.', flush=True)
