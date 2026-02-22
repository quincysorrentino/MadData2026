import { useState } from 'react'
import { useNavigate, useLocation, Navigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Header from '../components/Header/Header'
import DiagnosisImage from '../components/DiagnosisImage/DiagnosisImage'
import ChatWindow from '../components/ChatWindow/ChatWindow'
import ChatInput from '../components/ChatInput/ChatInput'
import { postChat } from '../api/chat'
import { useApp } from '../context/AppContext'
import type { ChatMessage } from '../types'
import styles from './ChatPage.module.css'

interface LocationState {
  fromUpload?: boolean
}

export default function ChatPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const {
    selectedBodyPartName,
    uploadedImageObjectUrl,
    diagnosis,
    boundingBox,
    resetSession,
  } = useApp()

  // Allow direct access if context has data (covers dev bypass) or came from upload flow
  const state = location.state as LocationState | null
  const hasData = uploadedImageObjectUrl && diagnosis
  if (!hasData) {
    return <Navigate to="/" replace />
  }
  void state

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSend(text: string) {
    const userMsg: ChatMessage = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setError(null)
    try {
      const result = await postChat(text)
      const assistantMsg: ChatMessage = { role: 'assistant', content: result.response }
      setMessages(prev => [...prev, assistantMsg])
    } catch {
      setError('Failed to get a response. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleStartOver() {
    resetSession()
    navigate('/', { replace: true })
  }

  return (
    <div className={styles.page}>
      <Header />
      <main className={styles.main}>
        <div className={styles.layout}>
          <div className={styles.topBar}>
            <div className={styles.breadcrumb}>
              <span>Body Area</span>
              <span className={styles.breadcrumbSep}>›</span>
              {selectedBodyPartName && (
                <span className={styles.bodyPartBadge}>{selectedBodyPartName}</span>
              )}
              <span className={styles.breadcrumbSep}>›</span>
              <span>Diagnosis</span>
            </div>
            <button className={styles.startOverBtn} onClick={handleStartOver} type="button">
              ← Start Over
            </button>
          </div>

          <div className={styles.columns}>
            {/* Left: uploaded image + follow-up chat */}
            <div className={styles.leftPanel}>
              <div className={styles.imageCard}>
                <span className={styles.imageLabel}>Uploaded Image</span>
                <DiagnosisImage
                  imageObjectUrl={uploadedImageObjectUrl}
                  boundingBox={boundingBox}
                />
              </div>

              <div className={styles.chatPanel}>
                <div className={styles.chatHeader}>
                  <div className={styles.chatTitle}>Follow-up Chat</div>
                  <div className={styles.chatSubtitle}>
                    Ask questions about your diagnosis
                  </div>
                </div>

                <div className={styles.chatBody}>
                  {error && (
                    <div className={`error-message ${styles.errorBanner}`}>{error}</div>
                  )}
                  <ChatWindow messages={messages} loading={loading} />
                  <ChatInput onSend={handleSend} disabled={loading} />
                </div>
              </div>
            </div>

            {/* Right: large diagnosis summary */}
            <div className={styles.rightPanel}>
              <div className={styles.diagnosisCard}>
                <div className={styles.diagnosisLabel}>AI Diagnosis Summary</div>
                <div className={styles.diagnosisText}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{diagnosis}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
