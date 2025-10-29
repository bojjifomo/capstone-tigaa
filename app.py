# app.py
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# =====================
# Setup
# =====================
st.set_page_config(page_title="CareerMatch AI — Resume RAG", page_icon="🧑‍💻")
st.title("CareerMatch AI — Resume RAG Agent")
st.caption("Capstone 3 • LangChain/LangGraph + Qdrant • Resume Dataset")

load_dotenv()
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GkYkSLuAAQZcvHjxacWT_yj6elpAiIeAfJ-bAmV8RPw", os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GkYkSLuAAQZcvHjxacWT_yj6elpAiIeAfJ-bAmV8RPw"))
OPENAI_API_KEY = st.secrets.get("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", os.getenv("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"))
COLLECTION_NAME = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "capstone"))
EMB_MODEL = st.secrets.get("EMB_MODEL", os.getenv("EMB_MODEL", "text-embedding-3-small"))
CHAT_MODEL = st.secrets.get("CHAT_MODEL", os.getenv("CHAT_MODEL", "gpt-4o-mini"))

if not (QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY):
    st.error("Missing one of required secrets: QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY")
    st.stop()

# =====================
# Optional: load dataset for HTML preview & category list
# =====================
resume_df = None
if os.path.exists("Resume.xlsx"):
    try:
        resume_df = pd.read_excel("Resume.xlsx")
    except Exception:
        resume_df = None
elif os.path.exists("Resume.csv"):
    try:
        resume_df = pd.read_csv("Resume.csv")
    except Exception:
        resume_df = None

id_to_html = {}
if isinstance(resume_df, pd.DataFrame) and "ID" in resume_df.columns:
    if "Resume_html" in resume_df.columns:
        for _, r in resume_df[["ID", "Resume_html"]].dropna(subset=["ID"]).iterrows():
            id_to_html[str(r["ID"])] = str(r.get("Resume_html", ""))

# =====================
# Models & VectorStore
# =====================
llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=EMB_MODEL, api_key=OPENAI_API_KEY)

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# =====================
# RAG Tool
# =====================
@tool
def search_resumes(query: str, k: int = 5, category: Optional[str] = None) -> List[dict]:
    """Cari resume paling relevan dengan query. Opsional filter kategori exact-match pada metadata."""
    results = qdrant.similarity_search(query, k=k)
    if category:
        results = [r for r in results if (r.metadata or {}).get("category", "").lower() == category.lower()]

    payload = []
    for r in results:
        meta = r.metadata or {}
        rid = meta.get("id", "-")
        cat = meta.get("category", "-")
        snippet = (r.page_content[:300] + "...") if len(r.page_content) > 300 else r.page_content
        payload.append({"id": rid, "category": cat, "snippet": snippet})
    return payload

TOOLS = [search_resumes]

# =====================
# Agent Prompt
# =====================
SYSTEM_PROMPT = (
    "You are an HR recruitment assistant specialized in resume search, summarization, and categorization.\n"
    "- Only answer questions related to resumes, jobs, or categories from this dataset.\n"
    "- When users ask for examples or similar resumes, ALWAYS call the `search_resumes` tool.\n"
    "- If a user pastes resume text, summarize key skills and suggest the closest category from:\n"
    "  HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness,\n"
    "  Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering,\n"
    "  Accountant, Construction, Public-Relations, Banking, Arts, Aviation.\n"
    "- Be concise. When presenting search results, show items with ID and Category.\n"
)

# =====================
# Agent Runner
# =====================
def run_agent(question: str, category_filter: Optional[str], k: int) -> Dict:
    user_msg = question
    if category_filter:
        user_msg += f"\n[search_hint] category={category_filter} k={k}"

    agent = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": user_msg}] })
    messages = result["messages"]

    answer = messages[-1].content

    # Token accounting
    total_input_tokens = 0
    total_output_tokens = 0
    for m in messages:
        meta = getattr(m, "response_metadata", {}) or {}
        if "usage_metadata" in meta:
            total_input_tokens += meta["usage_metadata"].get("input_tokens", 0)
            total_output_tokens += meta["usage_metadata"].get("output_tokens", 0)
        elif "token_usage" in meta:  # fallback older format
            tu = meta["token_usage"]
            total_input_tokens += tu.get("prompt_tokens", 0)
            total_output_tokens += tu.get("completion_tokens", 0)

    price_idr = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.60) / 1_000_000

    tool_messages = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_messages.append(m.content)

    return {
        "answer": answer,
        "price": price_idr,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
    }

# =====================
# Sidebar Controls
# =====================
with st.sidebar:
    st.header("🔧 Controls")
    k = st.slider("Top-K results", 3, 10, 5)
    categories_list = [
        "(none)", "HR", "Designer", "Information-Technology", "Teacher", "Advocate", "Business-Development",
        "Healthcare", "Fitness", "Agriculture", "BPO", "Sales", "Consultant", "Digital-Media", "Automobile",
        "Chef", "Finance", "Apparel", "Engineering", "Accountant", "Construction", "Public-Relations",
        "Banking", "Arts", "Aviation"
    ]
    category_filter = st.selectbox("Filter category (optional)", categories_list, index=0)
    if category_filter == "(none)":
        category_filter = None

    st.markdown("---")
    st.subheader("🧪 Classification Mode")
    paste_text = st.text_area("Paste raw resume text to summarize & suggest category (optional)", height=180)
    classify_btn = st.button("Classify Text")

# =====================
# Chat History
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =====================
# Classify Mode (no RAG call required)
# =====================
if classify_btn and paste_text.strip():
    with st.chat_message("assistant"):
        prompt = (
            "Summarize the following resume text in 3-5 bullet points (skills/experience), "
            "then suggest the closest category from the provided list.\n\n" + paste_text + "\n\n"
            "Return JSON with keys: summary (list[str]), suggested_category (str)."
        )
        resp = llm.invoke(prompt)
        st.markdown(resp.content)

# =====================
# Normal Chat (RAG via Agent)
# =====================
if user_q := st.chat_input("Ask about resumes, jobs, or categories..."):
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("assistant"):
        result = run_agent(user_q, category_filter=category_filter, k=k)
        st.markdown(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        # Debug panels
        with st.expander("**Tool Calls (Debug)**"):
            st.code("\n\n".join(map(str, result["tool_messages"])) or "(no tool calls)")
        with st.expander("**Usage Details**"):
            st.code(
                f"input tokens : {result['total_input_tokens']}\n"
                f"output tokens: {result['total_output_tokens']}\n"
                f"≈ price (IDR): {result['price']:.2f}"
            )

        # Optional: Preview HTML for any IDs mentioned in the answer
        if id_to_html:
            import re
            ids_found = set(re.findall(r"\b(?:ID[:\s]*)?(\d{1,6})\b", result["answer"]))
            if ids_found:
                with st.expander("Preview HTML (from local dataset)"):
                    for rid in sorted(ids_found):
                        html = id_to_html.get(str(rid))
                        if html:
                            st.markdown(f"**Resume ID {rid}**")
                            st.components.v1.html(html, height=400, scrolling=True)
