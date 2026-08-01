import { useId } from "react";
import type { RiskBand, WeeklyActivity } from "@/types";
import { riskColor } from "./primitives";

/**
 * ── The signature element ──────────────────────────────────────────────
 *
 * Every other part of this product is a fairly ordinary dashboard. This is
 * the piece that carries the idea: a student's weekly engagement drawn as a
 * ribbon, cut by a hard vertical rule at the prediction checkpoint.
 *
 * Everything left of the rule is what the model saw. Everything right of it
 * is greyed and hatched — time that has passed for the student but that the
 * prediction knows nothing about. That distinction is the entire premise of
 * an early-warning system, and showing it at every scale (16px tall in a
 * roster row, 160px tall on the detail page) keeps it in front of the advisor
 * rather than buried in a methodology note.
 */
export function EngagementRibbon({
  activity,
  band,
  height = 40,
  showAxis = false,
  className,
}: {
  activity: WeeklyActivity[];
  band: RiskBand;
  height?: number;
  showAxis?: boolean;
  className?: string;
}) {
  const uid = useId().replace(/:/g, "");
  const width = 100;
  const pad = 2;
  const color = riskColor(band);

  if (!activity.length) return null;

  const max = Math.max(...activity.map((a) => a.clicks), 1);
  const stepX = (width - pad * 2) / Math.max(1, activity.length - 1);
  const y = (clicks: number) => height - pad - (clicks / max) * (height - pad * 2);
  const x = (i: number) => pad + i * stepX;

  const before = activity.filter((a) => a.beforeCheckpoint);
  const checkpointX = before.length ? x(before.length - 1) : pad;

  const line = activity.map((a, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(2)} ${y(a.clicks).toFixed(2)}`).join(" ");
  const beforeLine = before.map((a, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(2)} ${y(a.clicks).toFixed(2)}`).join(" ");
  const beforeArea = `${beforeLine} L ${checkpointX.toFixed(2)} ${height - pad} L ${pad} ${height - pad} Z`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={className}
      style={{ height, width: "100%" }}
      role="img"
      aria-label={`Weekly course-site activity. ${before.length} weeks before the prediction checkpoint, ${
        activity.length - before.length
      } weeks after.`}
    >
      <defs>
        <linearGradient id={`fill-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
        <pattern id={`hatch-${uid}`} width="4" height="4" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <line x1="0" y="0" x2="0" y2="4" stroke="#C9D1CE" strokeWidth="1" />
        </pattern>
      </defs>

      {/* Time the model cannot see: hatched, deliberately inert. */}
      <rect
        x={checkpointX}
        y={0}
        width={width - checkpointX}
        height={height}
        fill={`url(#hatch-${uid})`}
        opacity="0.5"
      />

      <path d={beforeArea} fill={`url(#fill-${uid})`} />
      <path d={line} fill="none" stroke={color} strokeOpacity="0.28" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
      <path
        d={beforeLine}
        fill="none"
        stroke={color}
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />

      {/* The checkpoint itself — the moment the prediction was made. */}
      <line
        x1={checkpointX}
        y1={0}
        x2={checkpointX}
        y2={height}
        stroke="#16232E"
        strokeWidth="1.2"
        strokeDasharray="2.5 2"
        vectorEffect="non-scaling-stroke"
      />
      {showAxis && (
        <circle cx={checkpointX} cy={y(before[before.length - 1]?.clicks ?? 0)} r="2.6" fill="#16232E" />
      )}
    </svg>
  );
}
