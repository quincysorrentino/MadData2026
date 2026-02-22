import { useState } from 'react'
import { useNavigate, useLocation, Navigate } from 'react-router-dom'
import Header from '../components/Header/Header'
import ImageUploader from '../components/ImageUploader/ImageUploader'
import CameraCapture from '../components/CameraCapture/CameraCapture'
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner'
import { postDiagnose } from '../api/diagnose'
import { useApp } from '../context/AppContext'
import styles from './UploadPage.module.css'

type Mode = 'file' | 'camera'

interface LocationState {
  bodyPartName?: string
}

export default function UploadPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { setUploadedImage, setDiagnosisResult } = useApp()

  const state = location.state as LocationState | null
  if (!state?.bodyPartName) {
    return <Navigate to="/" replace />
  }

  const [mode, setMode] = useState<Mode>('file')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleFileSelect(file: File) {
    setSelectedFile(file)
    setError(null)
  }

  function handleCameraCapture(file: File) {
    setSelectedFile(file)
    setError(null)
  }

  function handleCameraCancel() {
    setMode('file')
  }

  async function handleSubmit() {
    if (!selectedFile) {
      setError('Please select or capture an image first.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      setUploadedImage(selectedFile)
      const result = await postDiagnose(selectedFile, state.bodyPartName)
      setDiagnosisResult(result.diagnosis, result.bounding_box)
      navigate('/chat', { state: { fromUpload: true } })
    } catch {
      setError('Failed to process the image. Make sure the backend is running and try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <Header />
      <main className={styles.main}>
        <div className={styles.content}>
          <div className={styles.breadcrumb}>
            <span>Body Area</span>
            <span className={styles.breadcrumbSep}>›</span>
            <span className={styles.bodyPartBadge}>{state.bodyPartName}</span>
            <span className={styles.breadcrumbSep}>›</span>
            <span>Upload Image</span>
          </div>

          <div className={styles.card}>
            <div className={styles.tabs}>
              <button
                className={`${styles.tab} ${mode === 'file' ? styles.tabActive : ''}`}
                onClick={() => setMode('file')}
                type="button"
              >
                Upload File
              </button>
              <button
                className={`${styles.tab} ${mode === 'camera' ? styles.tabActive : ''}`}
                onClick={() => setMode('camera')}
                type="button"
              >
                Use Camera
              </button>
            </div>

            <div className={styles.tabContent}>
              <h1 className={styles.heading}>
                {mode === 'file' ? 'Upload an Image' : 'Take a Photo'}
              </h1>
              <p className={styles.subheading}>
                {mode === 'file'
                  ? 'Select an image of the affected skin area from your device'
                  : 'Position the affected area in frame and capture a clear photo'}
              </p>

              {mode === 'file' ? (
                <ImageUploader
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                />
              ) : (
                <CameraCapture
                  onCapture={handleCameraCapture}
                  onCancel={handleCameraCancel}
                />
              )}
            </div>

            <div className={styles.submitRow}>
              {error && <div className={`error-message ${styles.error}`}>{error}</div>}

              {loading ? (
                <div className={styles.loadingWrapper}>
                  <LoadingSpinner small label="Analysing image..." />
                </div>
              ) : (
                <button
                  className={styles.submitBtn}
                  onClick={handleSubmit}
                  disabled={!selectedFile || loading}
                  type="button"
                >
                  Analyse Image
                </button>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
