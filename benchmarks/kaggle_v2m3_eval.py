import os, subprocess, sys, json

# 1) Repo
WORK = '/kaggle/working'
REPO = f'{WORK}/mnemonics'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', 'https://github.com/nakata-app/mnemonics.git', REPO], check=True)
subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-3'], check=True)

# 2) Bagimliliklar
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO,
                'sentence-transformers', 'numpy', 'adaptmem'], check=True)
print('Install OK')

# 3) Dataset
KAGGLE_DS = '/kaggle/input/mnemonics-lme/longmemeval_s_cleaned.json'
HF_DATA   = f'{WORK}/longmemeval_s.json'

if os.path.exists(KAGGLE_DS):
    DATA = KAGGLE_DS
    print(f'Kaggle dataset: {DATA}')
else:
    print('HuggingFace indiriliyor (~278 MB)...')
    subprocess.run(['wget', '-q', '--show-progress', '-O', HF_DATA,
        'https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s'], check=True)
    DATA = HF_DATA
print(f'Dataset: {os.path.getsize(DATA)/1e6:.1f} MB')

# 4) Env
os.makedirs(f'{WORK}/results', exist_ok=True)
env = os.environ.copy()
env['LME_DATA'] = DATA
env['MNEMONICS_RERANK_MODEL'] = 'BAAI/bge-reranker-v2-m3'

# 5) 100q eval
print('\n=== 100q basliyor ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '100', '--mode', 'rerank',
    '--augment-preferences', '--candidate-k', '50', '--seed', '42',
    '--out', f'{WORK}/results/lme100_v2m3.json',
    '--per-q-out', f'{WORK}/results/lme100_v2m3_perq.json',
], cwd=REPO, env=env, check=True)

# 6) 100q sonuc
r100 = json.load(open(f'{WORK}/results/lme100_v2m3.json'))['mnemonics_rerank']
print(f'\nbge-v2-m3 100q R@1={r100["R@1"]:.3f}  R@5={r100["R@5"]:.3f}  R@10={r100["R@10"]:.3f}')
print(f'Referans MiniLM: R@1=0.880 | Hedef MemPalace: R@1=0.920')

# 7) 500q eval
print('\n=== 500q basliyor ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '500', '--mode', 'rerank',
    '--augment-preferences', '--candidate-k', '50', '--seed', '42',
    '--out', f'{WORK}/results/lme500_v2m3.json',
    '--per-q-out', f'{WORK}/results/lme500_v2m3_perq.json',
], cwd=REPO, env=env, check=True)

# 8) Final
r500 = json.load(open(f'{WORK}/results/lme500_v2m3.json'))['mnemonics_rerank']
print(f'\n=== FINAL 500q ===')
print(f'R@1={r500["R@1"]:.3f}  R@5={r500["R@5"]:.3f}  R@10={r500["R@10"]:.3f}')
print(f'\nKarsilastirma:')
print(f'  Eski baseline:   R@1=0.846')
print(f'  MemPalace hedef: R@1=0.920')
print(f'\nBy type:')
for qt in sorted(r500['by_type']):
    b = r500['by_type'][qt]
    print(f'  {qt:28} n={b["n"]:3}  R@1={b["R@1"]:.3f}')
