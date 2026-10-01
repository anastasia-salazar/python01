import TaskItem from './TaskItem'

export default function TaskList({ tasks, totalTasks, onToggle, onDelete }) {
  if (tasks.length === 0) {
    return (
      <p className="card empty">
        {totalTasks === 0
          ? 'No tasks yet. Add your first task above!'
          : 'No tasks match your search or filter.'}
      </p>
    )
  }

  return (
    <ul className="task-list">
      {tasks.map((task) => (
        <TaskItem key={task.id} task={task} onToggle={onToggle} onDelete={onDelete} />
      ))}
    </ul>
  )
}
