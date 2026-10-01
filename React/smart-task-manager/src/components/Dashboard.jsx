export default function Dashboard({ stats }) {
  const { total, completed, pending, highPending, percentDone } = stats

  return (
    <section className="dashboard" aria-label="Task summary">
      <div className="stat card">
        <span className="stat-value">{total}</span>
        <span className="stat-label">Total tasks</span>
      </div>
      <div className="stat card">
        <span className="stat-value">{completed}</span>
        <span className="stat-label">Completed</span>
      </div>
      <div className="stat card">
        <span className="stat-value">{pending}</span>
        <span className="stat-label">Pending</span>
      </div>
      <div className="stat card">
        <span className="stat-value">{highPending}</span>
        <span className="stat-label">High priority left</span>
      </div>
      <div className="stat card progress-card">
        <div className="progress-header">
          <span className="stat-label">Progress</span>
          <span className="progress-percent">{percentDone}%</span>
        </div>
        <div
          className="progress-track"
          role="progressbar"
          aria-valuenow={percentDone}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Tasks completed"
        >
          <div className="progress-fill" style={{ width: `${percentDone}%` }} />
        </div>
      </div>
    </section>
  )
}
