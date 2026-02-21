import { useRef, useState, useEffect } from 'react'
import styles from './ImageUploader.module.css'

interface ImageUploaderProps {
  onFileSelect: (file: File) => void
  selectedFile: File | null
}

export default function ImageUploader({ onFileSelect, selectedFile }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile)
      setPreviewUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setPreviewUrl(null)
    }
  }, [selectedFile])

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) onFileSelect(file)
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setIsDragActive(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    setIsDragActive(false)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onFileSelect(file)
    }
  }

  function handleClick() {
    inputRef.current?.click()
  }

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff"
        className={styles.fileInput}
        onChange={handleFileChange}
      />

      {selectedFile && previewUrl ? (
        <div className={`${styles.dropzone} ${styles.dropzoneHasFile}`}>
          <div className={styles.preview}>
            <img src={previewUrl} alt="Selected" className={styles.previewImage} />
            <span className={styles.fileName}>{selectedFile.name}</span>
            <button className={styles.changeBtn} onClick={handleClick} type="button">
              Change image
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`${styles.dropzone} ${isDragActive ? styles.dropzoneActive : ''}`}
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && handleClick()}
          aria-label="Upload image"
        >
          <span className={styles.icon}>📷</span>
          <div className={styles.dropzoneText}>
            <span className={styles.primaryText}>
              {isDragActive ? 'Drop image here' : 'Click or drag an image here'}
            </span>
            <span className={styles.secondaryText}>
              Supports JPEG, PNG, WEBP, GIF, BMP, TIFF
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
