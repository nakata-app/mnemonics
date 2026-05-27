import os, subprocess, sys, json

# 1) Dizin
os.chdir('/content')

# 2) Repo
if not os.path.exists('/content/mnemonics'):
    subprocess.run(['git', 'clone', 'https://github.com/nakata-app/mnemonics.git', '/content/mnemonics'], check=True)

# 3) Pip
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '/content/mnemonics',
                'sentence-transformers', 'numpy', 'adaptmem'], check=True)

# 4) Dataset
DATA = '/content/longmemeval_s.json'
if not os.path.exists(DATA) or os.path.getsize(DATA) < 1_000_000:
    print('Dataset indiriliyor...')
    subprocess.run(['wget', '-q', '--show-progress', '-O', DATA,
        'https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_s.json'], check=True)
print(f'Dataset: {os.path.getsize(DATA)/1e6:.1f} MB')

# 5) 100q eval
os.makedirs('/content/results', exist_ok=True)
env = os.environ.copy()
env['LME_DATA'] = DATA
env['MNEMONICS_RERANK_MODEL'] = 'BAAI/bge-reranker-v2-m3'

print('\n=== 100q başlıyor ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '100', '--mode', 'rerank',
    '--augment-preferences', '--candidate-k', '50', '--seed', '42',
    '--out', '/content/results/lme100_v2m3.json',
    '--per-q-out', '/content/results/lme100_v2m3_perq.json',
], cwd='/content/mnemonics', env=env, check=True)

# 6) 100q sonuç
r100 = json.load(open('/content/results/lme100_v2m3.json'))['mnemonics_rerank']
print(f'\nbge-v2-m3 100q → R@1={r100["R@1"]:.3f}  R@5={r100["R@5"]:.3f}  R@10={r100["R@10"]:.3f}')
print(f'Referans MiniLM: R@1=0.880 | Hedef MemPalace: R@1=0.920')

# 7) 500q eval
print('\n=== 500q başlıyor ===')
subprocess.run([
    sys.executable, 'benchmarks/longmemeval_eval.py',
    '--n', '500', '--mode', 'rerank',
    '--augment-preferences', '--candidate-k', '50', '--seed', '42',
    '--out', '/content/results/lme500_v2m3.json',
    '--per-q-out', '/content/results/lme500_v2m3_perq.json',
], cwd='/content/mnemonics', env=env, check=True)

# 8) Final
r500 = json.load(open('/content/results/lme500_v2m3.json'))['mnemonics_rerank']
print(f'\n=== FINAL 500q ===')
print(f'R@1={r500["R@1"]:.3f}  R@5={r500["R@5"]:.3f}  R@10={r500["R@10"]:.3f}')
print(f'\nKarşılaştırma:')
print(f'  Eski baseline: R@1=0.846')
print(f'  MemPalace:     R@1=0.920')
print(f'\nBy type:')
for qt in sorted(r500['by_type']):
    b = r500['by_type'][qt]
    print(f'  {qt:28} n={b["n"]:3}  R@1={b["R@1"]:.3f}')
