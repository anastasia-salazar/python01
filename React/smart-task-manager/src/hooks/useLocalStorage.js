import { useEffect, useState } from 'react'

// Reads a saved value; falls back to `fallback` if nothing is saved or storage is
// unavailable (e.g. blocked in private browsing).
export function loadStored(key, fallback) {
  try {
    const saved = window.localStorage.getItem(key)
    return saved !== null ? JSON.parse(saved) : fallback
  } catch {
    return fallback
  }
}

// Custom hook: saves `value` in the browser every time it changes.
export function useSaveToStorage(key, value) {
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Storage full or blocked: the app keeps working, it just won't remember
    }
  }, [key, value])
}

// Custom hook: like useState, but the value survives a page refresh.
export function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => loadStored(key, initialValue))
  useSaveToStorage(key, value)
  return [value, setValue]
}
