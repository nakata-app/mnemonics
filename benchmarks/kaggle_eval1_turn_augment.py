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
HF_DATA   = f'{WORK}/longmemeval_s_cleaned.json'

if os.path.exists(KAGGLE_DS):
    DATA = KAGGLE_DS
    print(f'Kaggle dataset: {DATA}')
else:
    print('HuggingFace indiriliyor (~277 MB)...')
    subprocess.run(['wget', '-q', '--show-progress', '-O', HF_DATA,
        'https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_s_cleaned.json'],
        check=True)
    DATA = HF_DATA
print(f'Dataset: {os.path.getsize(DATA)/1e6:.1f} MB')

# 4) Env
os.makedirs(f'{WORK}/results', exist_ok=True)
env = os.environ.copy()
env['LME_DATA'] = DATA
# default encoder: all-MiniLM-L6-v2
# default CE:      cross-encoder/ms-marco-MiniLM-L-12-v2

# 5) 100q smoke
print('\n=== 100q smoke ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '100', '--mode', 'rerank',
    '--chunk-mode', 'turn',
    '--temporal-aware',
    '--augment-preferences',
    '--candidate-k', '50', '--seed', '42',
    '--out', f'{WORK}/results/lme100_eval1.json',
    '--per-q-out', f'{WORK}/results/lme100_eval1_perq.json',
], cwd=REPO, env=env, check=True)

r100 = json.load(open(f'{WORK}/results/lme100_eval1.json'))['mnemonics_rerank']
print(f'\nEval-1 100q → R@1={r100["R@1"]:.3f}  R@5={r100["R@5"]:.3f}  R@10={r100["R@10"]:.3f}')
print(f'Baseline (lme-0.954 config): R@1=0.954 R@5=0.988 R@10=0.994')

# 6) 500q
print('\n=== 500q basliyor ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '500', '--mode', 'rerank',
    '--chunk-mode', 'turn',
    '--temporal-aware',
    '--augment-preferences',
    '--candidate-k', '50', '--seed', '42',
    '--out', f'{WORK}/results/lme500_eval1.json',
    '--per-q-out', f'{WORK}/results/lme500_eval1_perq.json',
], cwd=REPO, env=env, check=True)

# 7) Sonuc
r500 = json.load(open(f'{WORK}/results/lme500_eval1.json'))['mnemonics_rerank']
print(f'\n=== EVAL-1 FINAL 500q ===')
print(f'Config: turn + temporal-aware + augment-preferences + default MiniLM CE')
print(f'R@1={r500["R@1"]:.3f}  R@5={r500["R@5"]:.3f}  R@10={r500["R@10"]:.3f}')
print(f'\nKarsilastirma:')
print(f'  lme-0.954 baseline: R@1=0.954  R@5=0.988  R@10=0.994')
print(f'  Hedef:              R@1=0.980  R@5=0.990  R@10=1.000')
delta = r500["R@1"] - 0.954
print(f'  Delta R@1: {delta:+.3f}')
print(f'\nBy type:')
for qt in sorted(r500['by_type']):
    b = r500['by_type'][qt]
    print(f'  {qt:30} n={b["n"]:3}  R@1={b["R@1"]:.3f}')
