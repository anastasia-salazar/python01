# Smart Task Manager

A small productivity app built with React 19 and Vite: a to-do list with a dashboard.

## Run it

```bash
npm install
npm run dev
```

Then open the URL it prints (in Codespaces, use the "Open in Browser" pop-up for port 5173).

## Features

- **Add tasks** with a title and a priority (High, Medium, Low); empty titles are rejected
- **Task list** showing each task's title, priority and status (Completed / Pending)
- **Mark as completed** with a checkbox (click again to set it back to Pending)
- **Delete** a task with the trash button, or **clear all completed** tasks at once
- **Search** tasks by title (case-insensitive)
- **Filter** to All, Pending or Completed tasks, with a count on each filter
- **Light / dark theme** switch that restyles the whole page
- **Dashboard** with totals, pending high-priority tasks and a progress bar
- Tasks and the chosen theme are **saved in the browser** and survive a refresh
- Pending tasks are listed first, highest priority first

## React hooks used

| Hook | Where | What it does |
|---|---|---|
| `useReducer` | `App.jsx`, `state/tasksReducer.js` | All task changes (add, toggle, delete, clear completed) go through one reducer |
| `useState` | `TaskForm.jsx`, `App.jsx` | Form inputs and validation message; search text; active filter |
| `useEffect` | `hooks/useLocalStorage.js`, `context/ThemeProvider.jsx` | Saves tasks and theme to the browser; applies the theme to the page |
| `useContext` | `hooks/useTheme.js` | Shares the theme with any component without passing props down |
| `useMemo` | `App.jsx`, `context/ThemeProvider.jsx` | Recomputes the filtered list and dashboard stats only when their inputs change |
| `useCallback` | `App.jsx`, `context/ThemeProvider.jsx` | Stable event handlers, so memoized task rows don't re-render needlessly |
| `useRef` | `TaskForm.jsx` | Puts the cursor back in the title box after adding a task |
| `useId` | `TaskForm.jsx`, `Toolbar.jsx` | Unique IDs linking labels to inputs (accessibility) |
| Custom hooks | `useLocalStorage`, `useSaveToStorage`, `useTheme` | Reusable logic built from the hooks above |

`TaskItem` is also wrapped in `React.memo`, so a row only re-renders when its own task changes.

## Project structure

```
src/
  App.jsx                    main component: state, search, filter, layout
  main.jsx                   entry point, wraps the app in ThemeProvider
  index.css                  styles; light and dark themes as CSS variables
  state/tasksReducer.js      the reducer and the starter tasks
  context/themeContext.js    the theme context object
  context/ThemeProvider.jsx  provides and saves the theme
  hooks/useLocalStorage.js   custom hooks for saving to the browser
  hooks/useTheme.js          custom hook to read/toggle the theme
  components/
    Dashboard.jsx            summary cards and progress bar
    TaskForm.jsx             add-task form
    Toolbar.jsx              search bar and filter buttons
    TaskList.jsx             the list, or an empty-state message
    TaskItem.jsx             one task row
    ThemeToggle.jsx          the light/dark switch
```
