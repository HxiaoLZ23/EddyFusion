type Props = {
  title: string;
  hint?: string;
  onClick: () => void;
  accent?: "slate" | "sky" | "amber" | "rose" | "emerald";
  /** tab=侧栏竖条（默认）；bar=底栏横条（L1 时占主区底部一行） */
  layout?: "tab" | "bar";
};

const acc: Record<string, { bg: string; border: string; color: string }> = {
  slate: { bg: "#f1f5f9", border: "#94a3b8", color: "#334155" },
  sky: { bg: "#e0f2fe", border: "#38bdf8", color: "#0369a1" },
  amber: { bg: "#fffbeb", border: "#fbbf24", color: "#92400e" },
  rose: { bg: "#fff1f2", border: "#fb7185", color: "#9f1239" },
  emerald: { bg: "#ecfdf5", border: "#34d399", color: "#047857" },
};

/** Level 1 时折叠为窄条，点击进入对应区块二级页 */
export function CollapsedStrip({ title, hint, onClick, accent = "slate", layout = "tab" }: Props) {
  const c = acc[accent] ?? acc.slate;
  return (
    <button
      type="button"
      className={`ocean-dashboard__collapsed-strip ocean-dashboard__collapsed-strip--${layout}`}
      style={{
        background: c.bg,
        border: `1px solid ${c.border}`,
        color: c.color,
      }}
      onClick={onClick}
      title={hint ?? title}
    >
      <span className="ocean-dashboard__collapsed-strip-title">{title}</span>
      <span className="ocean-dashboard__collapsed-strip-sub">点击进入详情</span>
    </button>
  );
}
