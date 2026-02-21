export interface BoundingBox {
  x: number
  y: number
  w: number
  h: number
}

export interface DiagnosisResult {
  diagnosis: string
  bounding_box: BoundingBox
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface BodyPartInfo {
  id: number
  name: string
}

export interface BodyPartResponse {
  success: boolean
  body_part_name: string
}

export interface ChatResponse {
  response: string
}
