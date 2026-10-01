import { useCallback, useEffect, useMemo } from 'react'
import { useLocalStorage } from '../hooks/useLocalStorage'
import { ThemeContext } from './themeContext'

// Start in the operating system's preferred theme, unless the user already picked one
const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'

export default function ThemeProvider({ children }) {
  const [theme, setTheme] = useLocalStorage('stm-theme', systemTheme)

  // Put the theme on <html data-theme="..."> so the CSS variables for that theme apply
  // to the entire page
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  const toggleTheme = useCallback(
    () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    [setTheme],
  )

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}
