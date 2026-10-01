import { useCallback, useMemo, useReducer, useState } from 'react'
import Dashboard from './components/Dashboard'
import TaskForm from './components/TaskForm'
import TaskList from './components/TaskList'
import ThemeToggle from './components/ThemeToggle'
import Toolbar from './components/Toolbar'
import { loadStored, useSaveToStorage } from './hooks/useLocalStorage'
import { STARTER_TASKS, tasksReducer } from './state/tasksReducer'

const PRIORITY_ORDER = { High: 0, Medium: 1, Low: 2 }

export default function App() {
  // useReducer: every change to the list goes through tasksReducer.
  // The third argument loads saved tasks once, on the first render.
  const [tasks, dispatch] = useReducer(tasksReducer, null, () =>
    loadStored('stm-tasks', STARTER_TASKS),
  )
  useSaveToStorage('stm-tasks', tasks) // useEffect inside: save after every change

  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('All')

  // useCallback: the same function objects on every render, so memoized TaskItems
  // don't re-render just because App did
  const addTask = useCallback((title, priority) => dispatch({ type: 'add', title, priority }), [])
  const toggleTask = useCallback((id) => dispatch({ type: 'toggle', id }), [])
  const deleteTask = useCallback((id) => dispatch({ type: 'delete', id }), [])
  const clearCompleted = useCallback(() => dispatch({ type: 'clearCompleted' }), [])

  // useMemo: only recalculate when the tasks change
  const stats = useMemo(() => {
    const completed = tasks.filter((t) => t.completed).length
    return {
      total: tasks.length,
      completed,
      pending: tasks.length - completed,
      highPending: tasks.filter((t) => !t.completed && t.priority === 'High').length,
      percentDone: tasks.length ? Math.round((completed / tasks.length) * 100) : 0,
    }
  }, [tasks])

  // useMemo: search + filter, then pending first and highest priority first
  const visibleTasks = useMemo(() => {
    const query = search.trim().toLowerCase()
    return tasks
      .filter((t) => t.title.toLowerCase().includes(query))
      .filter((t) =>
        filter === 'Completed' ? t.completed : filter === 'Pending' ? !t.completed : true,
      )
      .sort(
        (a, b) =>
          a.completed - b.completed ||
          PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority] ||
          b.createdAt - a.createdAt,
      )
  }, [tasks, search, filter])

  const filterCounts = { All: stats.total, Pending: stats.pending, Completed: stats.completed }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Smart Task Manager</h1>
          <p className="subtitle">Plan it, do it, check it off.</p>
        </div>
        <ThemeToggle />
      </header>

      <main>
        <Dashboard stats={stats} />
        <TaskForm onAdd={addTask} />

        <section className="tasks" aria-label="Tasks">
          <Toolbar
            search={search}
            onSearchChange={setSearch}
            filter={filter}
            onFilterChange={setFilter}
            counts={filterCounts}
          />
          <TaskList
            tasks={visibleTasks}
            totalTasks={tasks.length}
            onToggle={toggleTask}
            onDelete={deleteTask}
          />
          {stats.completed > 0 && (
            <div className="list-footer">
              <span>
                Showing {visibleTasks.length} of {stats.total}
              </span>
              <button type="button" className="link-button" onClick={clearCompleted}>
                Clear completed ({stats.completed})
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
