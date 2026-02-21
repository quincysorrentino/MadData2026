import type { ChatMessage } from '../../types'
import styles from './ChatBubble.module.css'

interface ChatBubbleProps {
  message: ChatMessage
}

export default function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === 'user'
  return (
    <div className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleAssistant}`}>
      <span className={styles.label}>{isUser ? 'You' : 'AI Assistant'}</span>
      <div className={styles.content}>{message.content}</div>
    </div>
  )
}
