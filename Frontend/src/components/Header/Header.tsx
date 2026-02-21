import { Link } from 'react-router-dom'
import styles from './Header.module.css'
import logo from '../../logo.png'

export default function Header() {
  return (
    <header className={styles.header}>
      <Link to="/" className={styles.logo}>
        <img src={logo} alt="DermaNet logo" className={styles.logoImg} />
      </Link>
      <div className={styles.textGroup}>
        <span className={styles.title}>DermaNet</span>
        <span className={styles.subtitle}>AI-Powered Dermatology Assistant</span>
      </div>
    </header>
  )
}
