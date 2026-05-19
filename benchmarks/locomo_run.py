import subprocess, sys, os, json, time, tempfile

# 1) PyPI versiyonu varsa sök (eskisi çakışıyor)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "mnemonics", "-y"],
               capture_output=True)

# 2) Repoyu çek (yoksa)
REPO = "/kaggle/working/mnemonics"
if not os.path.exists(f"{REPO}/mnemonics/__init__.py"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/nakata-app/mnemonics.git", REPO],
                   check=True)

# 3) Bağımlılıkları yükle
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers", "hnswlib", "rank-bm25", "tqdm"],
               check=True)

# 4) sys.modules'dan eski cache'i temizle, sonra path'e ekle
for k in list(sys.modules):
    if "mnemonics" in k:
        del sys.modules[k]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# 5) Import testi
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve
print("mnemonics import OK")

from tqdm import tqdm
from openai import OpenAI

# locomo10.json Kaggle'da "atakanakbaba/locomo" dataset'i olarak eklenmiş
_candidates = [
    "/kaggle/input/datasets/atakanakbaba/locomo/locomo10.json",
    "/kaggle/input/locomo/locomo10.json",
    "/kaggle/input/mnemonics-lme/locomo10.json",
]
LOCOMO_JSON = next((p for p in _candidates if os.path.exists(p)), None)
if LOCOMO_JSON is None:
    raise FileNotFoundError("locomo10.json bulunamadı, kontrol et: " + str(_candidates))

OUT = f"/kaggle/working/locomo_{time.strftime('%Y%m%d_%H%M%S')}.json"
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1")
data = json.load(open(LOCOMO_JSON))
results = {"mnemonics": []}

for conv in tqdm(data, desc="conv"):
    sid = conv["sample_id"]
    spk_a = conv["conversation"]["speaker_a"]
    spk_b = conv["conversation"]["speaker_b"]
    with tempfile.TemporaryDirectory() as td:
        store = Store(path=f"{td}/m.db")
        texts_a, meta_a, texts_b, meta_b = [], [], [], []
        for key in conv["conversation"]:
            if not key.startswith("session_") or key.endswith("_date_time"):
                continue
            dt = conv["conversation"].get(f"{key}_date_time", "")
            turns = conv["conversation"][key]
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                spk = turn.get("speaker", "")
                txt = turn.get("text", "")
                if not txt:
                    continue
                stamped = f"[{dt}] {spk}: {txt}"
                meta = {"ts": dt, "dia_id": turn.get("dia_id", "")}
                if spk == spk_a:
                    texts_a.append(stamped)
                    meta_a.append(meta)
                elif spk == spk_b:
                    texts_b.append(stamped)
                    meta_b.append(meta)
        if texts_a:
            ingest(texts=texts_a, store=store, ns=f"loc_{sid}_{spk_a}", meta=meta_a)
        if texts_b:
            ingest(texts=texts_b, store=store, ns=f"loc_{sid}_{spk_b}", meta=meta_b)
        for qa in tqdm(conv["qa"], desc=sid, leave=False):
            if str(qa.get("category")) == "5":
                continue
            q = qa["question"]
            gt = qa["answer"]
            hits_a = retrieve(query=q, store=store, ns=f"loc_{sid}_{spk_a}",
                              top_k=30, rerank=False)
            hits_b = retrieve(query=q, store=store, ns=f"loc_{sid}_{spk_b}",
                              top_k=30, rerank=False)
            mem_a = "\n".join(f"- {r['text']}" for r in hits_a.get("results", []))
            mem_b = "\n".join(f"- {r['text']}" for r in hits_b.get("results", []))
            prompt = (f"Memories {spk_a}:\n{mem_a}\n\n"
                      f"Memories {spk_b}:\n{mem_b}\n\n"
                      f"Question: {q}\nAnswer in 5 words max:")
            answer = "ERROR"
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                        temperature=0,
                    )
                    answer = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        time.sleep(10 * (attempt + 1))
                    else:
                        answer = f"ERROR: {e}"
                        break
            time.sleep(1)
            results["mnemonics"].append({
                "sample_id": sid,
                "question": q,
                "answer": gt,
                "response": answer,
                "category": str(qa.get("category")),
            })

json.dump(results, open(OUT, "w"), indent=2)
print(f"Done: {OUT} ({len(results['mnemonics'])} answers)")
