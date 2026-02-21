import { useState, useRef, useEffect } from 'react'
import styles from './BodySilhouette.module.css'

interface BodySilhouetteProps {
  onPartSelect: (partId: number) => void
  disabled?: boolean
}

interface BodyPartDef {
  id: number
  name: string
}

const BODY_PARTS: BodyPartDef[] = [
  { id: 1, name: 'Face / Head' },
  { id: 2, name: 'Neck' },
  { id: 3, name: 'Chest / Torso' },
  { id: 4, name: 'Back' },
  { id: 5, name: 'Arm / Hand' },
  { id: 6, name: 'Leg / Foot' },
]

// Never fully collapse to a line at 90°
const MIN_SCALE = 0.22

export default function BodySilhouette({ onPartSelect, disabled = false }: BodySilhouetteProps) {
  const [hoveredPartId, setHoveredPartId] = useState<number | null>(null)
  const [rotDeg,        setRotDeg]        = useState(0)
  const [isBackFacing,  setIsBackFacing]  = useState(false)

  const dragRef = useRef({ active: false, startX: 0, startRotDeg: 0, moved: false })
  const rotRef  = useRef(0)
  const wrapRef = useRef<HTMLDivElement>(null)

  // ── Drag + snap ────────────────────────────────────────────────
  useEffect(() => {
    function updateRot(val: number) {
      rotRef.current = val
      const norm = ((val % 360) + 360) % 360
      setRotDeg(val)
      setIsBackFacing(norm > 90 && norm < 270)
    }

    function snapTo(target: number) {
      const start = rotRef.current
      const diff  = ((target - start) % 360 + 540) % 360 - 180
      const dest  = start + diff
      const t0    = performance.now()
      const dur   = 380
      function step(now: number) {
        const t = Math.min((now - t0) / dur, 1)
        updateRot(start + diff * (1 - Math.pow(1 - t, 3)))
        if (t < 1) requestAnimationFrame(step)
        else updateRot(dest)
      }
      requestAnimationFrame(step)
    }

    function onMove(e: MouseEvent) {
      const d = dragRef.current
      if (!d.active) return
      const delta = e.clientX - d.startX
      if (Math.abs(delta) > 3) d.moved = true
      updateRot(d.startRotDeg + delta * 0.5)
    }
    function onTouchMove(e: TouchEvent) {
      const d = dragRef.current
      if (!d.active) return
      const delta = e.touches[0].clientX - d.startX
      if (Math.abs(delta) > 3) d.moved = true
      updateRot(d.startRotDeg + delta * 0.5)
    }
    function onUp() {
      const d = dragRef.current
      if (!d.active) return
      d.active = false
      if (wrapRef.current) wrapRef.current.style.cursor = 'grab'
      const norm = ((rotRef.current % 360) + 360) % 360
      snapTo(norm > 90 && norm < 270 ? 180 : 0)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onTouchMove, { passive: true })
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onUp)
    }
  }, [])

  function startDrag(clientX: number) {
    if (disabled) return
    dragRef.current = { active: true, startX: clientX, startRotDeg: rotRef.current, moved: false }
    if (wrapRef.current) wrapRef.current.style.cursor = 'grabbing'
  }

  // scaleX: compress at 90°, mirror when back-facing
  const cosR         = Math.cos((rotDeg * Math.PI) / 180)
  const displayScale = Math.max(Math.abs(cosR), MIN_SCALE)
  const scaleX       = cosR >= 0 ? displayScale : -displayScale

  // Torso reports id=4 when back-facing
  const torsoId = isBackFacing ? 4 : 3

  function handleEnter(id: number) { if (!disabled) setHoveredPartId(id) }
  function handleLeave()            { setHoveredPartId(null) }
  function handleClick(id: number)  { if (!dragRef.current.moved && !disabled) onPartSelect(id) }

  const partProps = (id: number) => ({
    className: `${styles.bodyPart}${hoveredPartId === id ? ` ${styles.bodyPartHovered}` : ''}`,
    onMouseEnter: () => handleEnter(id),
    onMouseLeave: handleLeave,
    onClick:      () => handleClick(id),
    style: disabled ? { cursor: 'not-allowed', opacity: 0.5 } as React.CSSProperties : undefined,
  })

  const hoveredName = BODY_PARTS.find(p => p.id === hoveredPartId)?.name

  return (
    <div className={styles.container}>
      <div className={styles.svgWrapper}>
        <div
          ref={wrapRef}
          className={styles.rotateWrapper}
          style={{ transform: `scaleX(${scaleX})` }}
          onMouseDown={e => startDrag(e.clientX)}
          onTouchStart={e => startDrag(e.touches[0].clientX)}
        >
          <svg
            viewBox="0 0 200 520"
            className={styles.svg}
            aria-label="Human body diagram — select an area"
          >
            {/* Silhouette outlines */}
            <ellipse cx="100" cy="52" rx="34" ry="44"                                                                       className={styles.bodyOutline} />
            <rect    x="87" y="93" width="26" height="26" rx="4"                                                            className={styles.bodyOutline} />
            <path    d="M58 119 Q58 115 63 115 L137 115 Q142 115 142 119 L148 265 Q148 270 143 270 L57 270 Q52 270 52 265 Z" className={styles.bodyOutline} />
            <path    d="M52 125 Q44 127 38 135 L20 290 Q18 298 24 300 L38 302 Q44 304 47 296 L62 150 Z"                      className={styles.bodyOutline} />
            <path    d="M148 125 Q156 127 162 135 L180 290 Q182 298 176 300 L162 302 Q156 304 153 296 L138 150 Z"             className={styles.bodyOutline} />
            <path    d="M57 268 L95 268 L92 450 Q91 458 85 458 L65 458 Q59 458 58 450 Z"                                     className={styles.bodyOutline} />
            <path    d="M105 268 L143 268 L142 450 Q141 458 135 458 L115 458 Q109 458 108 450 Z"                             className={styles.bodyOutline} />

            {/* Interactive zones */}
            <ellipse cx="100" cy="52" rx="34" ry="44"                                                                       {...partProps(1)} aria-label="Face / Head" />
            <rect    x="87" y="93" width="26" height="26" rx="4"                                                            {...partProps(2)} aria-label="Neck" />
            <path    d="M58 119 Q58 115 63 115 L137 115 Q142 115 142 119 L148 265 Q148 270 143 270 L57 270 Q52 270 52 265 Z" {...partProps(torsoId)} aria-label={isBackFacing ? 'Back' : 'Chest / Torso'} />
            <path    d="M52 125 Q44 127 38 135 L20 290 Q18 298 24 300 L38 302 Q44 304 47 296 L62 150 Z"                      {...partProps(5)} aria-label="Arm / Hand (left)" />
            <path    d="M148 125 Q156 127 162 135 L180 290 Q182 298 176 300 L162 302 Q156 304 153 296 L138 150 Z"             {...partProps(5)} aria-label="Arm / Hand (right)" />
            <path    d="M57 268 L95 268 L92 450 Q91 458 85 458 L65 458 Q59 458 58 450 Z"                                     {...partProps(6)} aria-label="Leg / Foot (left)" />
            <path    d="M105 268 L143 268 L142 450 Q141 458 135 458 L115 458 Q109 458 108 450 Z"                             {...partProps(6)} aria-label="Leg / Foot (right)" />
          </svg>
        </div>

        {/* Tooltip */}
        <div className={styles.tooltip}>
          {hoveredName && (
            <span className={styles.tooltipText}>{hoveredName}</span>
          )}
        </div>

        {/* Hint */}
        <p className={styles.hint}>
          {isBackFacing ? 'Back view — drag to flip front' : 'Drag left or right to rotate'}
        </p>
      </div>
    </div>
  )
}
