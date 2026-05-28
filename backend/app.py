from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

llm = ChatOllama(
    model="llama3.2",
    base_url="http://host.docker.internal:11434"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)

vectorstore = Chroma(
    collection_name="study_materials",
    embedding_function=embeddings,
    persist_directory="/chroma/chroma"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

DB_PATH = "/app/progress.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            question TEXT,
            answer TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/session", methods=["POST"])
def create_session():
    data = request.get_json()
    filename = data.get("filename", "Unknown")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (filename, created_at) VALUES (?, ?)",
              (filename, datetime.now().isoformat()))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id})

@app.route("/session/<int:session_id>/history", methods=["GET"])
def get_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, answer, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
              (session_id,))
    rows = c.fetchall()
    conn.close()
    return jsonify([{"question": r[0], "answer": r[1], "created_at": r[2]} for r in rows])

@app.route("/sessions", methods=["GET"])
def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT s.id, s.filename, s.created_at, COUNT(m.id) as question_count
        FROM sessions s
        LEFT JOIN messages m ON s.id = m.session_id
        GROUP BY s.id
        ORDER BY s.created_at DESC
        LIMIT 10
    """)
    rows = c.fetchall()
    conn.close()
    return jsonify([{
        "id": r[0],
        "filename": r[1],
        "created_at": r[2],
        "question_count": r[3]
    } for r in rows])

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif filename.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(tmp_path)
        else:
            return jsonify({"error": "Unsupported file type. Use PDF or PPTX."}), 400

        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vectorstore.add_documents(chunks)

        return jsonify({
            "message": f"Successfully ingested {filename}",
            "chunks": len(chunks)
        })

    finally:
        os.unlink(tmp_path)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    relevant_chunks = vectorstore.similarity_search(user_message, k=8)

    if relevant_chunks:
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        system_prompt = f"""You are a helpful study tutor analyzing a student's uploaded course material.

Here are the most relevant sections from their uploaded document:

{context}

Using this material, answer the student's question thoroughly and specifically. 
- If the document is an exam or study guide, identify the course, topics, and key concepts explicitly.
- If you can see question types or subjects covered, list them clearly.
- Be specific — reference actual content from the material, not general knowledge.
- If something isn't in the material, say so."""
    else:
        system_prompt = "You are a helpful study tutor. No study materials have been uploaded yet. Let the user know they can upload a PDF or PowerPoint to get started."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    answer = response.content

    if session_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO messages (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                  (session_id, user_message, answer, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    return jsonify({"response": answer})

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    topic = data.get("topic", "the uploaded material")
    session_id = data.get("session_id")

    relevant_chunks = vectorstore.similarity_search(topic, k=8)

    if not relevant_chunks:
        return jsonify({"error": "No study materials uploaded yet."}), 400

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    system_prompt = f"""You are a study tutor generating practice questions from a student's course material.

Here are relevant sections from their uploaded document:

{context}

Generate exactly 5 practice questions based strictly on this material.
Format your response exactly like this, with no extra text before or after:

Q1: [question]
A1: [answer]

Q2: [question]
A2: [answer]

Q3: [question]
A3: [answer]

Q4: [question]
A4: [answer]

Q5: [question]
A5: [answer]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate 5 practice questions about: {topic}")
    ]

    response = llm.invoke(messages)
    raw = response.content

    questions = []
    lines = raw.strip().split("\n")
    current_q = None
    current_a = None

    for line in lines:
        line = line.strip()
        if line.startswith("Q") and ":" in line and line[1].isdigit():
            if current_q and current_a:
                questions.append({"question": current_q, "answer": current_a})
            current_q = line.split(":", 1)[1].strip()
            current_a = None
        elif line.startswith("A") and ":" in line and line[1].isdigit():
            current_a = line.split(":", 1)[1].strip()

    if current_q and current_a:
        questions.append({"question": current_q, "answer": current_a})

    if session_id and questions:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for q in questions:
            c.execute("INSERT INTO messages (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                      (session_id, q["question"], q["answer"], datetime.now().isoformat()))
        conn.commit()
        conn.close()

    return jsonify({"questions": questions})

@app.route("/check-answer", methods=["POST"])
def check_answer():
    data = request.get_json()
    question = data.get("question")
    reference_answer = data.get("reference_answer")
    user_answer = data.get("user_answer")

    if not all([question, reference_answer, user_answer]):
        return jsonify({"error": "Missing fields"}), 400

    system_prompt = """You are a fair and encouraging study tutor evaluating a student's answer.
You understand that there are often multiple valid ways to answer a question correctly.
Be generous — if the student demonstrates understanding of the core concept, that counts.

Respond in exactly this format:
VERDICT: Correct / Partially Correct / Incorrect
FEEDBACK: [2-3 sentences explaining why, what they got right, and what they missed if anything]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Question: {question}

Reference Answer: {reference_answer}

Student's Answer: {user_answer}

Evaluate the student's answer.""")
    ]

    response = llm.invoke(messages)
    raw = response.content

    verdict = "Unknown"
    feedback = raw

    for line in raw.strip().split("\n"):
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    return jsonify({"verdict": verdict, "feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
