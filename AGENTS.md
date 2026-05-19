# Mnemonics, Agent Integration Guide

Mnemonics gives any AI agent persistent, decay-aware memory in under 10 lines.
Pick the integration that matches your stack.

---

## How it works

```
agent turn N  →  retrieve(query)  →  inject top-k into context
              →  ingest(turn text) →  store for future turns
```

Each memory has a **tier** (pinned / default / ambient) that controls how fast it fades.
Tier 0 = never decays. Tier 1 = 90-day half-life. Tier 2 = 14-day half-life.
Retrieval score = `cosine × decay_factor`, so stale memories rank lower automatically.

---

## LangChain

Drop `MnemonicsMemory` in as a replacement for `ConversationBufferMemory`:

```python
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.messages import HumanMessage, AIMessage
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve


class MnemonicsMemory(BaseChatMemory):
    store: Store = None
    ns: str = "default"
    top_k: int = 5

    def __init__(self, store_path="~/.mnemonics", ns="default", top_k=5):
        super().__init__()
        object.__setattr__(self, "store", Store(store_path))
        object.__setattr__(self, "ns", ns)
        object.__setattr__(self, "top_k", top_k)

    @property
    def memory_variables(self):
        return ["memory"]

    def load_memory_variables(self, inputs):
        query = inputs.get("input", "")
        hits = retrieve(query, self.store, top_k=self.top_k, ns=self.ns)
        context = "\n".join(r["text"] for r in hits["results"])
        return {"memory": context}

    def save_context(self, inputs, outputs):
        turn = f"Human: {inputs.get('input','')}\nAI: {outputs.get('output','')}"
        ingest([turn], self.store, ns=self.ns)

    def clear(self):
        pass
```

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = MnemonicsMemory(ns="my-agent")
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
chain.predict(input="What did we discuss about the deployment last week?")
```

---

## CrewAI

Wrap mnemonics as a tool that crew members can call:

```python
from crewai_tools import BaseTool
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

_store = Store("~/.mnemonics")

class MemoryIngestTool(BaseTool):
    name: str = "memory_ingest"
    description: str = "Store a fact or observation in long-term memory."

    def _run(self, text: str) -> str:
        ingest([text], _store)
        return "stored"

class MemoryRetrieveTool(BaseTool):
    name: str = "memory_retrieve"
    description: str = "Recall relevant facts from long-term memory."

    def _run(self, query: str) -> str:
        hits = retrieve(query, _store, top_k=5)
        return "\n".join(f"- {r['text']}" for r in hits["results"])
```

```python
from crewai import Agent

analyst = Agent(
    role="Research Analyst",
    goal="Answer questions using long-term memory",
    tools=[MemoryRetrieveTool(), MemoryIngestTool()],
)
```

---

## AutoGen

Register as function tools on any `AssistantAgent`:

```python
import autogen
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

store = Store("~/.mnemonics")

def memory_store(text: str) -> str:
    ingest([text], store)
    return "stored"

def memory_recall(query: str) -> str:
    hits = retrieve(query, store, top_k=5)
    return "\n".join(r["text"] for r in hits["results"])

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "functions": [
            {"name": "memory_store",  "description": "Store a fact.",   "parameters": {"type": "object", "properties": {"text":  {"type": "string"}}, "required": ["text"]}},
            {"name": "memory_recall", "description": "Recall relevant facts.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        ],
        "config_list": [{"model": "gpt-4o", "api_key": "..."}],
    },
    function_map={"memory_store": memory_store, "memory_recall": memory_recall},
)
```

---

## LlamaIndex

Plug in as a custom memory module:

```python
from llama_index.core.memory import BaseMemory
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve
from llama_index.core.llms import ChatMessage


class MnemonicsMemory(BaseMemory):
    def __init__(self, store_path="~/.mnemonics", ns="default"):
        self._store = Store(store_path)
        self._ns = ns

    def get(self, input: str = "", **kwargs):
        hits = retrieve(input, self._store, top_k=5, ns=self._ns)
        if not hits["results"]:
            return []
        context = "\n".join(r["text"] for r in hits["results"])
        return [ChatMessage(role="system", content=f"Relevant memory:\n{context}")]

    def put(self, message: ChatMessage) -> None:
        if message.role in ("user", "assistant"):
            ingest([str(message.content)], self._store, ns=self._ns)

    def set(self, messages):
        for m in messages:
            self.put(m)

    def reset(self):
        pass
```

---

## REST / any framework

Start the server once, call it from anywhere:

```bash
mnem serve --port 7810
```

```python
import httpx

BASE = "http://127.0.0.1:7810"

def remember(text: str, ns: str = "default"):
    httpx.post(f"{BASE}/ingest", json={"texts": [text], "ns": ns})

def recall(query: str, ns: str = "default", top_k: int = 5):
    r = httpx.post(f"{BASE}/retrieve", json={"query": query, "top_k": top_k, "ns": ns})
    return [row["text"] for row in r.json()["results"]]
```

Works from any language, curl, Go, Rust, Node, same two endpoints.

---

## Multi-agent namespaces

Isolate each agent's memory with a namespace. Agents never see each other's memories unless you explicitly share a namespace:

```python
planner_memory  = Store("~/.mnemonics")  # ns="planner"
executor_memory = Store("~/.mnemonics")  # ns="executor"
shared_memory   = Store("~/.mnemonics")  # ns="shared"

# Planner writes a decision to shared memory
ingest(["Deploy target: staging, not prod."], planner_memory, ns="shared", tier=0)

# Executor reads it
recall = retrieve("deploy target", executor_memory, ns="shared")
```

Tier 0 (`pin`) is recommended for cross-agent decisions, it never decays regardless of access frequency.

---

## MCP (Claude Code, Cursor, Metis)

```bash
mnem mcp   # starts the MCP server
```

Add to your MCP config:

```json
{
  "mcpServers": {
    "mnemonics": {
      "command": "mnem",
      "args": ["mcp"]
    }
  }
}
```

Tools available: `mnemonics_ingest`, `mnemonics_retrieve`, `mnemonics_forget`, `mnemonics_pin`, `mnemonics_tier`, `mnemonics_gc`, `mnemonics_stats`.

---

## Benchmarks

Evaluated on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (500-question memory retrieval benchmark):

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| MemPalace entity-graph baseline | 0.354 | n/a | n/a |
| **Mnemonics (no rerank)** | **0.880** | **0.900** | **0.900** |
| **Mnemonics (CE rerank)** | **0.920+** | n/a | n/a |

CE rerank = AdaptMem cross-encoder reranking over the candidate band.
Full 500-question results in progress.
