import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <span className={styles.logoIcon}>⬡</span>
      </div>
      <div className={styles.textGroup}>
        <span className={styles.title}>DermaNet</span>
        <span className={styles.subtitle}>AI-Powered Dermatology Assistant</span>
      </div>
    </header>
  )
}
