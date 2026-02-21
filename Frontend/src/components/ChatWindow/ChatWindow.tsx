import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../../types'
import ChatBubble from '../ChatBubble/ChatBubble'
import styles from './ChatWindow.module.css'

interface ChatWindowProps {
  messages: ChatMessage[]
  loading: boolean
}

export default function ChatWindow({ messages, loading }: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className={styles.window}>
      {messages.map((msg, i) => (
        <ChatBubble key={i} message={msg} />
      ))}
      {loading && (
        <div className={styles.typingBubble}>
          <div className={styles.dot} />
          <div className={styles.dot} />
          <div className={styles.dot} />
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
