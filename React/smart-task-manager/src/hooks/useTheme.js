import { useContext } from 'react'
import { ThemeContext } from '../context/themeContext'

// Custom hook so any component can read or change the theme
export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used inside <ThemeProvider>')
  return context
}
