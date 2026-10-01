import { useTheme } from '../hooks/useTheme'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <button
      type="button"
      role="switch"
      aria-checked={isDark}
      className="theme-toggle"
      onClick={toggleTheme}
    >
      <span className="toggle-track" aria-hidden="true">
        <span className="toggle-thumb" />
      </span>
      {isDark ? 'Dark mode' : 'Light mode'}
    </button>
  )
}
