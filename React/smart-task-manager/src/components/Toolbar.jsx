import { useId } from 'react'

const FILTERS = ['All', 'Pending', 'Completed']

export default function Toolbar({ search, onSearchChange, filter, onFilterChange, counts }) {
  const searchId = useId()

  return (
    <div className="toolbar">
      <div className="search">
        <label htmlFor={searchId} className="visually-hidden">
          Search tasks
        </label>
        <input
          id={searchId}
          type="search"
          placeholder="Search tasks by title..."
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
        />
      </div>
      <div className="filters" role="group" aria-label="Filter tasks">
        {FILTERS.map((name) => (
          <button
            key={name}
            type="button"
            className={filter === name ? 'filter active' : 'filter'}
            aria-pressed={filter === name}
            onClick={() => onFilterChange(name)}
          >
            {name} <span className="count">{counts[name]}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
