import subprocess, sys, os, json, time, tempfile
from tqdm import tqdm
from openai import OpenAI

subprocess.run(["pip", "uninstall", "mnemonics", "-y"], capture_output=True)
subprocess.run(["pip", "install", "hnswlib", "-q"], capture_output=True)
sys.path.insert(0, "/kaggle/working/mnemonics")

from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-4cd36d293a7f4020bfd0ead28f08b01a")
BUNDLE = "/kaggle/input/datasets/atakanakbaba/mnemonics-kaggle-bundle"
OUT = f"/kaggle/working/locomo_{time.strftime('%Y%m%d_%H%M%S')}.json"
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
data = json.load(open(f"{BUNDLE}/locomo10.json"))
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
            hits_a = retrieve(query=q, store=store, ns=f"loc_{sid}_{spk_a}", top_k=30, rerank=False)
            hits_b = retrieve(query=q, store=store, ns=f"loc_{sid}_{spk_b}", top_k=30, rerank=False)
            mem_a = "\n".join(f"- {r['text']}" for r in hits_a.get("results", []))
            mem_b = "\n".join(f"- {r['text']}" for r in hits_b.get("results", []))
            prompt = f"Memories {spk_a}:\n{mem_a}\n\nMemories {spk_b}:\n{mem_b}\n\nQuestion: {q}\nAnswer in 5 words max:"
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
