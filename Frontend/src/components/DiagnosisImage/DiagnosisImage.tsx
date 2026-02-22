import { useRef, useState, useEffect } from 'react'
import type { BoundingBox } from '../../types'
import styles from './DiagnosisImage.module.css'

interface DiagnosisImageProps {
  imageObjectUrl: string
  boundingBox: BoundingBox | null
}

interface BoxStyle {
  left: number
  top: number
  width: number
  height: number
}

export default function DiagnosisImage({ imageObjectUrl, boundingBox }: DiagnosisImageProps) {
  const imgRef = useRef<HTMLImageElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const [boxStyle, setBoxStyle] = useState<BoxStyle | null>(null)

  function computeBox() {
    const img = imgRef.current
    if (!img || !img.complete || img.naturalWidth === 0 || !boundingBox) {
      setBoxStyle(null)
      return
    }
    const scaleX = img.clientWidth / img.naturalWidth
    const scaleY = img.clientHeight / img.naturalHeight
    setBoxStyle({
      left: boundingBox.x * scaleX,
      top: boundingBox.y * scaleY,
      width: boundingBox.w * scaleX,
      height: boundingBox.h * scaleY,
    })
  }

  useEffect(() => {
    const wrapper = wrapperRef.current
    if (!wrapper) return
    const observer = new ResizeObserver(computeBox)
    observer.observe(wrapper)
    return () => observer.disconnect()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boundingBox])

  return (
    <div ref={wrapperRef} className={styles.wrapper}>
      <img
        ref={imgRef}
        src={imageObjectUrl}
        alt="Uploaded skin image"
        className={styles.image}
        onLoad={computeBox}
      />
      {boxStyle && (
        <div
          className={styles.boundingBox}
          style={{
            left: boxStyle.left,
            top: boxStyle.top,
            width: boxStyle.width,
            height: boxStyle.height,
          }}
        >
          <span className={styles.boundingBoxLabel}>Lesion</span>
        </div>
      )}
    </div>
  )
}
