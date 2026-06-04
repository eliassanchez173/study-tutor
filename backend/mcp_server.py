import asyncio
import sqlite3
import chromadb
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

chroma_client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)

vectorstore = Chroma(
    collection_name="study_materials",
    embedding_function=embeddings,
    client=chroma_client,
)

DB_PATH = "/Users/sanchez/study-tutor/backend/progress.db"

server = Server("study-tutor")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_materials",
            description="Search the student's uploaded study materials for relevant content. Use this when answering questions about course material.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="generate_questions",
            description="Generate practice exam questions based on the student's uploaded study materials.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or concept to generate questions about"
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "Number of questions to generate (default 5)",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        ),
        types.Tool(
            name="check_answer",
            description="Evaluate a student's answer to a practice question and provide feedback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The practice question"
                    },
                    "reference_answer": {
                        "type": "string",
                        "description": "The reference or model answer"
                    },
                    "student_answer": {
                        "type": "string",
                        "description": "The student's answer to evaluate"
                    }
                },
                "required": ["question", "reference_answer", "student_answer"]
            }
        ),
        types.Tool(
            name="get_session_history",
            description="Retrieve the student's past study sessions and question history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent sessions to retrieve (default 5)",
                        "default": 5
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_materials":
        query = arguments["query"]
        chunks = vectorstore.similarity_search(query, k=6)
        if not chunks:
            return [types.TextContent(type="text", text="No study materials found. Please upload a PDF or PPTX first.")]
        result = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])
        return [types.TextContent(type="text", text=f"Relevant content from study materials:\n\n{result}")]

    elif name == "generate_questions":
        topic = arguments["topic"]
        num = arguments.get("num_questions", 5)
        chunks = vectorstore.similarity_search(topic, k=8)
        if not chunks:
            return [types.TextContent(type="text", text="No study materials found. Please upload a PDF or PPTX first.")]
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        return [types.TextContent(type="text", text=f"Generate {num} practice questions about '{topic}' based on this material:\n\n{context}")]

    elif name == "check_answer":
        question = arguments["question"]
        reference = arguments["reference_answer"]
        student = arguments["student_answer"]
        return [types.TextContent(type="text", text=f"""Evaluate this student answer:

Question: {question}

Reference Answer: {reference}

Student Answer: {student}

Provide a verdict (Correct/Partially Correct/Incorrect) and specific feedback.""")]

    elif name == "get_session_history":
        limit = arguments.get("limit", 5)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT s.filename, s.created_at, COUNT(m.id) as question_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        conn.close()
        if not rows:
            return [types.TextContent(type="text", text="No study sessions found yet.")]
        result = "\n".join([f"- {r[0]}: {r[2]} questions on {r[1][:10]}" for r in rows])
        return [types.TextContent(type="text", text=f"Recent study sessions:\n{result}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())