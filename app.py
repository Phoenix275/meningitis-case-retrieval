import os, time, json
import streamlit as st
from openai import OpenAI
import faiss, numpy as np

# Init clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
index = faiss.read_index("index.faiss")
with open("data.jsonl","r") as f:
    records = [json.loads(line) for line in f]

# Streamlit layout
st.title("Meningitis Case-Similarity & Summary Demo")
st.markdown(
    "Enter a brief patient description and see the top-5 similar published case reports, "
    "each with a one-sentence treatment/outcome summary."
)

query = st.text_input("Patient profile", 
    "18 y/o with fever, neck stiffness, CSF neutrophils >1000"
)

if st.button("Retrieve & Summarize"):
    # Embed the query
    q_resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    q_emb = np.array(q_resp.data[0].embedding, dtype="float32").reshape(1,-1)
    time.sleep(0.5)

    # Search index
    D, I = index.search(q_emb, 5)

    # Display results
    for rank, idx in enumerate(I[0], start=1):
        rec = records[idx]
        title = rec["prompt"].split("\n",1)[0].replace("Title: ","")
        pmid = rec["metadata"]["pmid"]

        # Summarize
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":
                "Summarize the treatment and outcome of this case report in one sentence:\n" 
                + rec["prompt"]
            }],
            temperature=0.0,
            max_tokens=60
        )
        summary = resp.choices[0].message.content.strip()
        time.sleep(1)

        st.markdown(f"**{rank}. PMID {pmid}** — *{title}*")
        st.write(summary)