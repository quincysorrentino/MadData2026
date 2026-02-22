import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header/Header'
import BodySilhouette from '../components/BodySilhouette/BodySilhouette'
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner'
import { useApp } from '../context/AppContext'
import styles from './HomePage.module.css'

const BODY_PART_NAMES: Record<number, string> = {
  1: 'Face / Head',
  2: 'Neck',
  3: 'Chest / Torso',
  4: 'Back',
  5: 'Arm / Hand',
  6: 'Leg / Foot',
}

export default function HomePage() {
  const navigate = useNavigate()
  const { setBodyPart } = useApp()
  const [loading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handlePartSelect(partId: number) {
    const bodyPartName = BODY_PART_NAMES[partId] ?? 'Unknown Area'
    setBodyPart(partId, bodyPartName)
    setError(null)
    navigate('/upload', { state: { bodyPartName } })
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
        </div>
      </main>
    </div>
  )
}
