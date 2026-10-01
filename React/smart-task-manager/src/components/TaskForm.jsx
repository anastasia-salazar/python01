import { useId, useRef, useState } from 'react'
import { PRIORITIES } from '../state/tasksReducer'

export default function TaskForm({ onAdd }) {
  const [title, setTitle] = useState('')
  const [priority, setPriority] = useState('Medium')
  const [error, setError] = useState('')
  const titleInput = useRef(null)
  const titleId = useId()
  const priorityId = useId()

  function handleSubmit(event) {
    event.preventDefault()
    const trimmed = title.trim()
    if (!trimmed) {
      setError('Please enter a task title.')
      titleInput.current.focus()
      return
    }
    onAdd(trimmed, priority)
    setTitle('')
    setError('')
    titleInput.current.focus() // ready to type the next task straight away
  }

  return (
    <form className="card task-form" onSubmit={handleSubmit} noValidate>
      <div className="field grow">
        <label htmlFor={titleId}>Task</label>
        <input
          id={titleId}
          ref={titleInput}
          type="text"
          placeholder="What needs to be done?"
          value={title}
          maxLength={120}
          onChange={(event) => {
            setTitle(event.target.value)
            if (error) setError('')
          }}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? `${titleId}-error` : undefined}
        />
      </div>
      <div className="field">
        <label htmlFor={priorityId}>Priority</label>
        <select id={priorityId} value={priority} onChange={(e) => setPriority(e.target.value)}>
          {PRIORITIES.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>
      </div>
      <button type="submit" className="primary">
        Add task
      </button>
      {error && (
        <p className="form-error" id={`${titleId}-error`} role="alert">
          {error}
        </p>
      )}
    </form>
  )
}
