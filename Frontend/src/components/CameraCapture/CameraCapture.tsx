import { useRef, useState, useEffect } from 'react'
import styles from './CameraCapture.module.css'

interface CameraCaptureProps {
  onCapture: (file: File) => void
  onCancel: () => void
}

export default function CameraCapture({ onCapture, onCancel }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)

  // Wire the stream to the video element after it renders into the DOM
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream
      videoRef.current.play().catch(() => {/* autoplay policy — playsInline handles mobile */})
    }
  }, [stream])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stream?.getTracks().forEach(t => t.stop())
      if (capturedUrl) URL.revokeObjectURL(capturedUrl)
    }
  }, [stream, capturedUrl])

  async function startCamera() {
    setCameraError(null)
    try {
      // Try environment (rear) camera first; fall back to any available camera
      let mediaStream: MediaStream
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
      } catch {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      }
      setStream(mediaStream)
    } catch {
      setCameraError('Camera access denied. Please allow camera permissions and try again.')
    }
  }

  function captureFrame() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    // Guard: video must have actual dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0)

    // Stop stream AFTER drawing the frame
    const currentStream = stream
    setStream(null)
    currentStream?.getTracks().forEach(t => t.stop())

    canvas.toBlob(blob => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      setCapturedUrl(url)
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
      onCapture(file)
    }, 'image/jpeg', 0.92)
  }

  function retake() {
    if (capturedUrl) URL.revokeObjectURL(capturedUrl)
    setCapturedUrl(null)
    startCamera()
  }

  function stopCamera() {
    stream?.getTracks().forEach(t => t.stop())
    setStream(null)
    onCancel()
  }

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {!stream && !capturedUrl && (
        <>
          <button className={styles.startBtn} onClick={startCamera} type="button">
            <span className={styles.startBtnIcon}>📹</span>
            <span>Open Camera</span>
            <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Uses your device camera to take a photo
            </span>
          </button>
          {cameraError && <p className={styles.errorText}>{cameraError}</p>}
        </>
      )}

      {stream && (
        <>
          <div className={styles.videoWrapper}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={styles.video}
            />
          </div>
          <div className={styles.controls}>
            <button className={styles.captureBtn} onClick={captureFrame} type="button">
              Capture Photo
            </button>
            <button className={styles.cancelBtn} onClick={stopCamera} type="button">
              Cancel
            </button>
          </div>
        </>
      )}

      {capturedUrl && (
        <>
          <img src={capturedUrl} alt="Captured" className={styles.capturedPreview} />
          <div className={styles.capturedActions}>
            <button className={styles.retakeBtn} onClick={retake} type="button">
              Retake
            </button>
          </div>
        </>
      )}
    </div>
  )
}
