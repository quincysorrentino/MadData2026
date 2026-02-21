import styles from './LoadingSpinner.module.css'

interface LoadingSpinnerProps {
  label?: string
  small?: boolean
}

export default function LoadingSpinner({ label, small = false }: LoadingSpinnerProps) {
  return (
    <div className={styles.wrapper}>
      <div className={`${styles.spinner} ${small ? styles.spinnerSmall : ''}`} />
      {label && <span className={styles.label}>{label}</span>}
    </div>
  )
}
