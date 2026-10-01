import { memo } from 'react'

// memo: a task only re-renders when its own data changes, which works because App
// passes stable handler functions (useCallback)
function TaskItem({ task, onToggle, onDelete }) {
  const checkboxId = `task-${task.id}`

  return (
    <li className={task.completed ? 'task completed' : 'task'}>
      <input
        id={checkboxId}
        type="checkbox"
        checked={task.completed}
        onChange={() => onToggle(task.id)}
      />
      <label htmlFor={checkboxId} className="task-title">
        {task.title}
      </label>
      <span className={`badge priority-${task.priority.toLowerCase()}`}>{task.priority}</span>
      <span className={task.completed ? 'status done' : 'status'}>
        {task.completed ? 'Completed' : 'Pending'}
      </span>
      <button
        type="button"
        className="delete"
        onClick={() => onDelete(task.id)}
        aria-label={`Delete "${task.title}"`}
        title="Delete task"
      >
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path
            d="M9 3h6m-9 4h12m-1 0-.7 11.2a2 2 0 0 1-2 1.8H9.7a2 2 0 0 1-2-1.8L7 7m3 4v5m4-5v5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </li>
  )
}

export default memo(TaskItem)
