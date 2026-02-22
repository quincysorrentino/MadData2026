import type { ChatResponse } from '../types'

const CHAT_TIMEOUT_MS = 60_000

export async function postChat(message: string): Promise<ChatResponse> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS)

  let res: Response
  try {
    res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Chat request timed out. The local LLM may be overloaded.')
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }

  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.json() as Promise<ChatResponse>
}
