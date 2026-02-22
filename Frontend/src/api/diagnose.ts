import type { DiagnosisResult } from '../types'

const DIAGNOSE_TIMEOUT_MS = 90_000

export async function postDiagnose(imageFile: File, bodyPartName?: string): Promise<DiagnosisResult> {
  const form = new FormData()
  form.append('image', imageFile)
  if (bodyPartName) {
    form.append('body_part_name', bodyPartName)
  }

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), DIAGNOSE_TIMEOUT_MS)

  let res: Response
  try {
    res = await fetch('/api/diagnose', {
      method: 'POST',
      body: form,
      signal: controller.signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Diagnosis request timed out. The local LLM may be overloaded.')
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }

  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.json() as Promise<DiagnosisResult>
}
