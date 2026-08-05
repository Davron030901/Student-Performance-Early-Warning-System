import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Info, TrendingDown, TrendingUp } from "lucide-react";
import { EngagementRibbon } from "@/components/ui/EngagementRibbon";
import { Card, CardHeader, ErrorState, RiskChip, Skeleton, riskColor } from "@/components/ui/primitives";
import { useStudentDetail } from "@/lib/api/hooks";
import type { RiskFactor } from "@/types";

export function StudentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: s, isLoading, isError, refetch } = useStudentDetail(id);

  return (
    <div className="mx-auto max-w-5xl px-4 py-7 sm:px-6 lg:px-10 lg:py-10">
      <Link
        to="/students"
        className="mb-5 inline-flex min-h-[44px] items-center gap-1.5 text-sm font-semibold text-ink-muted hover:text-brand"
      >
        <ArrowLeft size={16} aria-hidden /> All students
      </Link>

      {isError ? (
        <Card>
          <ErrorState what="this student" onRetry={() => refetch()} />
        </Card>
      ) : isLoading || !s ? (
        <DetailSkeleton />
      ) : (
        <div className="animate-fade-up space-y-5">
          {/* Identity + score */}
          <Card className="p-5 sm:p-6">
            <div className="flex flex-wrap items-start justify-between gap-5">
              <div>
                <p className="mb-1.5 font-mono text-eyebrow uppercase text-ink-muted">
                  {s.id} · {s.courseCode}
                </p>
                <h1 className="font-display text-display-md font-bold tracking-tight text-ink">{s.name}</h1>
                <div className="mt-3">
                  <RiskChip band={s.riskBand} />
                </div>
              </div>

              <div className="text-right">
                <p className="font-mono text-eyebrow uppercase text-ink-muted">Risk score</p>
                <p
                  className="nums font-display text-display-lg font-bold leading-none"
                  style={{ color: riskColor(s.riskBand) }}
                >
                  {/* Clamped to 1–99. A model with ~78% recall has no business
                      telling an advisor a student has a "100% chance of not
                      passing" — that is a claim of certainty about a person
                      that the evidence cannot support, and it invites treating
                      the score as a verdict rather than a prompt to look. The
                      same applies at the bottom: "0%" would read as a guarantee. */}
                  {Math.min(99, Math.max(1, Math.round(s.riskScore * 100)))}
                  <span className="text-2xl">%</span>
                </p>
                <p className="mt-1 text-xs text-ink-muted">estimated chance of not passing</p>
              </div>
            </div>
          </Card>

          {/* The signature ribbon, at full size */}
          <Card>
            <CardHeader title="Engagement through the term" eyebrow="Course-site activity" />
            <div className="px-5 py-5">
              <EngagementRibbon activity={s.activity} band={s.riskBand} height={150} showAxis />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-ink-muted">
                <span className="flex items-center gap-2">
                  <span className="inline-block h-0.5 w-6 rounded" style={{ background: riskColor(s.riskBand) }} aria-hidden />
                  Weekly clicks the model could see
                </span>
                <span className="flex items-center gap-2">
                  <span className="inline-block h-3.5 w-0 border-l border-dashed border-ink" aria-hidden />
                  Prediction made here — {s.checkpointUsed}
                </span>
                <span className="flex items-center gap-2">
                  <span className="inline-block h-3 w-4 bg-line" aria-hidden />
                  After the checkpoint — not used
                </span>
              </div>
            </div>
          </Card>

          <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
            {/* Why */}
            <Card>
              <CardHeader title="What's driving this" eyebrow="Contributing factors" />
              <ul className="divide-y divide-line">
                {s.topFactors.map((f: RiskFactor) => {
                  const raises = f.impact > 0;
                  const width = Math.min(100, Math.abs(f.impact) * 240);
                  return (
                    <li key={f.text} className="px-5 py-3.5">
                      <div className="flex items-start gap-3">
                        {raises ? (
                          <TrendingUp size={16} className="mt-0.5 shrink-0 text-risk-high" strokeWidth={2.2} aria-hidden />
                        ) : (
                          <TrendingDown size={16} className="mt-0.5 shrink-0 text-risk-low" strokeWidth={2.2} aria-hidden />
                        )}
                        <div className="min-w-0 flex-1">
                          <p className="text-[15px] leading-snug text-ink">{f.text}</p>
                          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-line/70">
                            <div
                              className="h-full rounded-full transition-[width] duration-300"
                              style={{
                                width: `${width}%`,
                                background: raises ? "#A8443A" : "#3D6E8F",
                              }}
                            />
                          </div>
                          <p className="mt-1 text-xs text-ink-muted">
                            {raises ? "Raises" : "Lowers"} the score
                          </p>
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <p className="flex items-start gap-2 border-t border-line px-5 py-3 text-xs leading-relaxed text-ink-muted">
                <Info size={14} className="mt-0.5 shrink-0" aria-hidden />
                These are behavioural signals only. Demographic information is deliberately excluded from
                explanations, so a check-in is never prompted by who a student is.
              </p>
            </Card>

            {/* Numbers */}
            <Card>
              <CardHeader title="At the checkpoint" eyebrow="Recorded activity" />
              <dl className="divide-y divide-line">
                {[
                  ["Assessments submitted", `${s.submittedCount} of ${s.expectedCount}`],
                  ["Average early score", s.avgEarlyScore === null ? "No submissions yet" : `${s.avgEarlyScore}/100`],
                  ["Submitted on time", `${Math.round(s.onTimeRate * 100)}%`],
                  ["Total clicks", s.totalClicks.toLocaleString()],
                  ["Active days", String(s.activeDays)],
                  ["Last signed in", `${s.lastActiveDaysAgo} days ago`],
                  [
                    "Registered",
                    s.registeredDay < 0 ? `${Math.abs(s.registeredDay)} days before start` : `${s.registeredDay} days after start`,
                  ],
                ].map(([label, value]) => (
                  <div key={label} className="flex items-baseline justify-between gap-4 px-5 py-3">
                    <dt className="text-sm text-ink-muted">{label}</dt>
                    <dd className="nums text-right font-mono text-sm font-medium text-ink">{value}</dd>
                  </div>
                ))}
              </dl>
              <p className="border-t border-line px-5 py-3 font-mono text-xs text-ink-muted">
                Scored by {s.modelVersion} at {s.checkpointUsed}
              </p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-5">
      <Card className="p-6">
        <Skeleton className="h-3 w-28" />
        <Skeleton className="mt-3 h-9 w-64" />
        <Skeleton className="mt-4 h-7 w-36" />
      </Card>
      <Card className="p-5">
        <Skeleton className="h-[150px] w-full" />
      </Card>
      <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        <Card className="p-5">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="py-3">
              <Skeleton className="h-4 w-56" />
              <Skeleton className="mt-2 h-1.5 w-full" />
            </div>
          ))}
        </Card>
        <Card className="p-5">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="flex justify-between py-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-4 w-16" />
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}
