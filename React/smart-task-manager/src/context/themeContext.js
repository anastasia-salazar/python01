import { createContext } from 'react'

// Holds { theme, toggleTheme }; provided by <ThemeProvider>, read with useTheme()
export const ThemeContext = createContext(null)
