import type { BodyPartResponse } from '../types'

export async function postBodyPart(bodyPartId: number): Promise<BodyPartResponse> {
  const res = await fetch('/api/body-part', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ body_part_id: bodyPartId }),
  })
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.json() as Promise<BodyPartResponse>
}
