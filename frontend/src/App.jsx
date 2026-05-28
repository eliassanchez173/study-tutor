import { useState, useRef, useEffect } from "react"
import axios from "axios"
import ReactMarkdown from "react-markdown"
import { useDropzone } from "react-dropzone"
import "./App.css"

const API = "http://localhost:5001"

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [uploading, setUploading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [activeTab, setActiveTab] = useState("upload")
  const [questions, setQuestions] = useState([])
  const [quizTopic, setQuizTopic] = useState("")
  const [generatingQuiz, setGeneratingQuiz] = useState(false)
  const [revealed, setRevealed] = useState({})
  const [answerInputs, setAnswerInputs] = useState({})
  const [results, setResults] = useState({})
  const [checkingAnswer, setCheckingAnswer] = useState({})
  const bottomRef = useRef(null)

  useEffect(() => { fetchSessions() }, [])

  const fetchSessions = async () => {
    try {
      const res = await axios.get(`${API}/sessions`)
      setSessions(res.data)
    } catch {}
  }

  const loadSession = async (session) => {
    try {
      const res = await axios.get(`${API}/session/${session.id}/history`)
      setSessionId(session.id)
      setUploadedFile(session.filename)
      setMessages(res.data.flatMap(entry => [
        { role: "user", content: entry.question },
        { role: "assistant", content: entry.answer }
      ]))
      setActiveTab("upload")
    } catch {}
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"]
    },
    multiple: false,
    onDrop: async (files) => {
      const file = files[0]
      if (!file) return
      setUploading(true)
      const formData = new FormData()
      formData.append("file", file)
      try {
        const uploadRes = await axios.post(`${API}/upload`, formData)
        const sessionRes = await axios.post(`${API}/session`, { filename: file.name })
        setSessionId(sessionRes.data.session_id)
        setUploadedFile(file.name)
        setMessages([{
          role: "system",
          content: `✓ Uploaded **${file.name}** — ${uploadRes.data.chunks} chunks indexed. Ask me anything about it.`
        }])
        setQuestions([])
        fetchSessions()
      } catch {
        setMessages(prev => [...prev, { role: "system", content: "Upload failed. Please try again." }])
      } finally {
        setUploading(false)
      }
    }
  })

  const sendMessage = async () => {
    if (!input.trim()) return
    const userMessage = input.trim()
    setInput("")
    setMessages(prev => [...prev, { role: "user", content: userMessage }])
    setLoading(true)
    try {
      const res = await axios.post(`${API}/chat`, { message: userMessage, session_id: sessionId })
      setMessages(prev => [...prev, { role: "assistant", content: res.data.response }])
      fetchSessions()
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "Something went wrong. Please try again." }])
    } finally {
      setLoading(false)
      bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }

  const generateQuestions = async () => {
    setGeneratingQuiz(true)
    setQuestions([])
    setRevealed({})
    try {
      const res = await axios.post(`${API}/generate-questions`, {
        topic: quizTopic || "the uploaded material",
        session_id: sessionId
      })
      setQuestions(res.data.questions)
      fetchSessions()
    } catch {
      alert("Failed to generate questions. Make sure a file is uploaded.")
    } finally {
      setGeneratingQuiz(false)
    }
  }

  const toggleReveal = (i) => {
    setRevealed(prev => ({ ...prev, [i]: !prev[i] }))
  }

  const clearSession = () => {
    setUploadedFile(null)
    setMessages([])
    setSessionId(null)
    setQuestions([])
    setRevealed({})
  }
  const checkAnswer = async (i, question, referenceAnswer) => {
    if (!answerInputs[i]?.trim()) return
    setCheckingAnswer(prev => ({ ...prev, [i]: true }))
    try {
      const res = await axios.post(`${API}/check-answer`, {
        question,
        reference_answer: referenceAnswer,
        user_answer: answerInputs[i]
      })
      setResults(prev => ({ ...prev, [i]: res.data }))
    } catch {
      setResults(prev => ({ ...prev, [i]: { verdict: "Error", feedback: "Something went wrong." } }))
    } finally {
      setCheckingAnswer(prev => ({ ...prev, [i]: false }))
    }
  }
  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="logo">Study Tutor</h1>

        <div className="tabs">
          <button className={`tab ${activeTab === "upload" ? "active" : ""}`} onClick={() => setActiveTab("upload")}>Upload</button>
          <button className={`tab ${activeTab === "history" ? "active" : ""}`} onClick={() => setActiveTab("history")}>History</button>
        </div>

        {activeTab === "upload" && (
          <>
            <p className="subtitle">Upload your course materials and ask anything.</p>
            <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""} ${uploading ? "uploading" : ""}`}>
              <input {...getInputProps()} />
              {uploading
                ? <p>Uploading...</p>
                : uploadedFile
                ? <p className="uploaded">✓ {uploadedFile}</p>
                : <p>{isDragActive ? "Drop it here" : "Drag & drop a PDF or PPTX, or click to browse"}</p>
              }
            </div>
            {uploadedFile && (
              <button className="clear-btn" onClick={clearSession}>Clear</button>
            )}
          </>
        )}

        {activeTab === "history" && (
          <div className="history">
            {sessions.length === 0
              ? <p className="no-history">No sessions yet.</p>
              : sessions.map(s => (
                <div key={s.id} className="session-card" onClick={() => loadSession(s)}>
                  <p className="session-file">{s.filename}</p>
                  <p className="session-meta">{s.question_count} questions · {new Date(s.created_at).toLocaleDateString()}</p>
                </div>
              ))
            }
          </div>
        )}
      </aside>

      <main className="chat">
        <div className="chat-tabs">
          <button className={`chat-tab ${questions.length === 0 ? "active" : ""}`} onClick={() => setQuestions([])}>Chat</button>
          <button className={`chat-tab ${questions.length > 0 ? "active" : ""}`} onClick={() => questions.length > 0 && setQuestions(questions)}>Practice Quiz</button>
        </div>

        {questions.length === 0 ? (
          <>
            <div className="messages">
              {messages.length === 0 && <div className="empty">Upload a file to get started.</div>}
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ))}
              {loading && <div className="message assistant"><span className="typing">Thinking...</span></div>}
              <div ref={bottomRef} />
            </div>
            <div className="quiz-trigger">
              <input
                className="input"
                value={quizTopic}
                onChange={e => setQuizTopic(e.target.value)}
                placeholder="Topic to quiz on (optional)..."
                disabled={generatingQuiz || !uploadedFile}
              />
              <button className="quiz-btn" onClick={generateQuestions} disabled={generatingQuiz || !uploadedFile}>
                {generatingQuiz ? "Generating..." : "Generate Quiz"}
              </button>
            </div>
            <div className="input-row">
              <input
                className="input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendMessage()}
                placeholder="Ask a question about your materials..."
                disabled={loading}
              />
              <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
                Send
              </button>
            </div>
          </>
        ) : (
          <div className="quiz">
            <div className="quiz-header">
              <h2 className="quiz-title">Practice Quiz</h2>
              <button className="clear-btn" onClick={() => { setQuestions([]); setRevealed({}) }}>Back to chat</button>
            </div>
            <div className="questions">
            {questions.map((q, i) => (
  <div key={i} className="question-card">
    <p className="question-text">Q{i + 1}: {q.question}</p>

    {!results[i] ? (
      <div className="answer-input-row">
        <input
          className="answer-input"
          value={answerInputs[i] || ""}
          onChange={e => setAnswerInputs(prev => ({ ...prev, [i]: e.target.value }))}
          onKeyDown={e => e.key === "Enter" && checkAnswer(i, q.question, q.answer)}
          placeholder="Type your answer..."
          disabled={checkingAnswer[i]}
        />
        <button
          className="submit-answer-btn"
          onClick={() => checkAnswer(i, q.question, q.answer)}
          disabled={checkingAnswer[i] || !answerInputs[i]?.trim()}
        >
          {checkingAnswer[i] ? "Checking..." : "Submit"}
        </button>
      </div>
    ) : (
      <div className="result">
        <p className={`verdict verdict-${results[i].verdict.toLowerCase().replace(" ", "-")}`}>
          {results[i].verdict}
        </p>
        <p className="feedback">{results[i].feedback}</p>
        <p className="your-answer">Your answer: {answerInputs[i]}</p>
        <p className="reference-answer">Reference answer: {q.answer}</p>
        <button className="retry-btn" onClick={() => {
          setResults(prev => { const n = {...prev}; delete n[i]; return n })
          setAnswerInputs(prev => { const n = {...prev}; delete n[i]; return n })
        }}>Try again</button>
      </div>
    )}
  </div>
))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}