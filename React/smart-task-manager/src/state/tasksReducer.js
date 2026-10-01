// All changes to the task list go through this reducer, used with useReducer.
// Each action describes what happened; the reducer returns the new list without
// modifying the old one.

export const PRIORITIES = ['High', 'Medium', 'Low']

export function tasksReducer(tasks, action) {
  switch (action.type) {
    case 'add':
      return [
        {
          id: crypto.randomUUID(),
          title: action.title,
          priority: action.priority,
          completed: false,
          createdAt: Date.now(),
        },
        ...tasks,
      ]
    case 'toggle':
      return tasks.map((task) =>
        task.id === action.id ? { ...task, completed: !task.completed } : task,
      )
    case 'delete':
      return tasks.filter((task) => task.id !== action.id)
    case 'clearCompleted':
      return tasks.filter((task) => !task.completed)
    default:
      throw new Error(`Unknown action: ${action.type}`)
  }
}

// A few tasks so the app isn't empty the first time it opens
export const STARTER_TASKS = [
  { title: 'Finish the React assignment', priority: 'High', completed: false },
  { title: 'Review LangGraph notes', priority: 'Medium', completed: true },
  { title: 'Water the plants', priority: 'Low', completed: false },
].map((task, i) => ({ ...task, id: `starter-${i}`, createdAt: Date.now() - i }))
