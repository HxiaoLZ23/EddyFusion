import type { JobRecord } from "../adapters/jobAdapter";

type Props = {
  job: JobRecord | null;
  label?: string;
};

export function JobProgressBar({ job, label = "任务进度" }: Props) {
  if (!job) return null;
  const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
  const color =
    job.status === "failed" ? "#b91c1c" : job.status === "done" ? "#166534" : "#0369a1";
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#475569" }}>
        <span>
          {label} · {job.phase ?? job.status}
        </span>
        <span>{pct}%</span>
      </div>
      <div
        style={{
          height: 8,
          borderRadius: 4,
          background: "#e2e8f0",
          marginTop: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            transition: "width 0.3s ease",
          }}
        />
      </div>
      {job.message && (
        <p style={{ margin: "4px 0 0", fontSize: 11, color: "#64748b" }}>{job.message}</p>
      )}
    </div>
  );
}
