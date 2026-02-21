import type { DiagnosisResult } from '../types'

export async function postDiagnose(imageFile: File): Promise<DiagnosisResult> {
  const form = new FormData()
  form.append('image', imageFile)
  const res = await fetch('/api/diagnose', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.json() as Promise<DiagnosisResult>
}
