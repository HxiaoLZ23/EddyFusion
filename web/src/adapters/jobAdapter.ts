import { apiUrl, formatApiFetchError } from "./apiBase";

export type JobType = "eddy_dual_mp4" | "windwave_forecast";

export type JobRecord = {
  id: string;
  type: JobType;
  status: "pending" | "running" | "done" | "failed";
  progress: number;
  phase?: string;
  message?: string;
  result?: Record<string, unknown>;
  error?: string | null;
};

export async function postCreateJob(body: {
  type: JobType;
  nc_path: string;
  fps?: number;
  max_frames?: number;
  top_k?: number;
}): Promise<{ job_id: string }> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/api/jobs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error(formatApiFetchError(e));
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    throw new Error(typeof d === "string" ? d : res.statusText);
  }
  const data = (await res.json()) as { job_id: string };
  return { job_id: data.job_id };
}

export async function fetchJobStatus(jobId: string): Promise<JobRecord> {
  const res = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(jobId)}`));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const d = (err as { detail?: unknown }).detail;
    throw new Error(typeof d === "string" ? d : res.statusText);
  }
  const data = (await res.json()) as { job: JobRecord };
  return data.job;
}

/** 轮询直至 done/failed 或超时 */
export async function pollJobUntilDone(
  jobId: string,
  onTick?: (job: JobRecord) => void,
  intervalMs = 1200,
  timeoutMs = 600_000,
): Promise<JobRecord> {
  const t0 = Date.now();
  for (;;) {
    const job = await fetchJobStatus(jobId);
    onTick?.(job);
    if (job.status === "done" || job.status === "failed") {
      if (job.status === "failed") {
        throw new Error(job.error || job.message || "任务失败");
      }
      return job;
    }
    if (Date.now() - t0 > timeoutMs) {
      throw new Error("任务轮询超时");
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
