import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header/Header'
import BodySilhouette from '../components/BodySilhouette/BodySilhouette'
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner'
import { postBodyPart } from '../api/bodyPart'
import { useApp } from '../context/AppContext'
import styles from './HomePage.module.css'

const MOCK_DIAGNOSIS =
  `[DEV MODE] This is a placeholder diagnosis response.\n\nThe AI has detected a suspicious lesion in the selected area. Based on the classification, this appears consistent with a benign seborrheic keratosis. However, please consult a licensed dermatologist for any medical concerns.\n\nKey observations:\n• Irregular border detected\n• Asymmetric pigmentation\n• Diameter estimated at 8mm`

export default function HomePage() {
  const navigate = useNavigate()
  const { setBodyPart, setUploadedImage, setDiagnosisResult } = useApp()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleDevBypass() {
    // Create a tiny placeholder image blob so the chat page has something to display
    const canvas = document.createElement('canvas')
    canvas.width = 400
    canvas.height = 300
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#1e3a5f'
    ctx.fillRect(0, 0, 400, 300)
    ctx.fillStyle = '#4a90d9'
    ctx.font = 'bold 18px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Dev Mode — Placeholder Image', 200, 150)
    canvas.toBlob(blob => {
      if (!blob) return
      const file = new File([blob], 'dev-placeholder.png', { type: 'image/png' })
      setBodyPart(1, 'Face / Head (Dev)')
      setUploadedImage(file)
      setDiagnosisResult(MOCK_DIAGNOSIS, { x: 120, y: 80, w: 100, h: 80 })
      navigate('/chat')
    }, 'image/png')
  }

  async function handlePartSelect(partId: number) {
    setLoading(true)
    setError(null)
    try {
      const result = await postBodyPart(partId)
      if (result.success) {
        setBodyPart(partId, result.body_part_name)
        navigate('/upload', { state: { bodyPartName: result.body_part_name } })
      } else {
        setError('Failed to select body part. Please try again.')
      }
    } catch {
      setError('Could not connect to the server. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <Header />
      <main className={styles.main}>
        <div className={styles.content}>
          <div className={styles.intro}>
            <h1 className={styles.heading}>Select the Affected Body Area</h1>
            <p className={styles.subheading}>
              Hover over a region and click to begin your assessment
            </p>
          </div>

          {loading ? (
            <div className={styles.loadingWrapper}>
              <LoadingSpinner label="Loading..." />
            </div>
          ) : (
            <BodySilhouette onPartSelect={handlePartSelect} disabled={loading} />
          )}

          {error && (
            <div className={`error-message ${styles.error}`}>{error}</div>
          )}

          <div className={styles.devBypass}>
            <button className={styles.devBtn} onClick={handleDevBypass} type="button">
              Dev: Preview Chat Page →
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
