import os
from typing import List
import chromadb
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 向量数据库
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# ====== 工具函数 ======
def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


# ====== Agent 1：知识解析 ======
def parse_agent(text: str) -> str:
    prompt = f"""
你是一个知识解析助手，请将以下内容结构化为要点：
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ====== Agent 2：知识存储 ======
def store_knowledge(text: str):
    embedding = get_embedding(text)
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )


# ====== Agent 3：检索 ======
def retrieve(query: str, top_k=3):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results["documents"][0]


# ====== Agent 4：推理问答 ======
def qa_agent(query: str):
    docs = retrieve(query)
    context = "\n".join(docs)

    prompt = f"""
基于以下知识回答问题：

知识：
{context}

问题：
{query}
"""
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ====== Agent 5：总结 ======
def summary_agent():
    all_docs = collection.get()["documents"]
    text = "\n".join(all_docs)

    prompt = f"""
请总结以下知识，并给出关键结论：

{text}
"""
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ====== CLI ======
def main():
    print("🧠 Personal Knowledge Agent")
    print("1. 添加知识")
    print("2. 提问")
    print("3. 总结")
    print("4. 退出")

    while True:
        choice = input("\n选择功能: ")

        if choice == "1":
            text = input("输入知识内容: ")
            parsed = parse_agent(text)
            store_knowledge(parsed)
            print("✅ 已存储")

        elif choice == "2":
            q = input("你的问题: ")
            answer = qa_agent(q)
            print("\n💡 答案:")
            print(answer)

        elif choice == "3":
            summary = summary_agent()
            print("\n📊 总结:")
            print(summary)

        elif choice == "4":
            break


if __name__ == "__main__":
    main()
