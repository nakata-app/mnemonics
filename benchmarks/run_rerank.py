import re, pathlib, os, sys

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DATA = '/content/drive/MyDrive/longmemeval_s_cleaned.json'
p = pathlib.Path('/content/mnemonics/benchmarks/longmemeval_eval.py')
p.write_text(re.sub(r'DATA = Path\([^)]+\)', 'DATA = Path("' + DATA + '")', p.read_text()))

os.makedirs('/content/results', exist_ok=True)
os.chdir('/content/mnemonics')

sys.argv = [
    'x',
    '--n', '500',
    '--mode', 'rerank',
    '--augment-preferences',
    '--augment-assistant-facts',
    '--candidate-k', '50',
    '--seed', '42',
    '--out', '/content/results/lme500_facts_rerank.json',
    '--per-q-out', '/content/results/lme500_facts_rerank_perq.json',
]
exec(open('benchmarks/longmemeval_eval.py').read())
