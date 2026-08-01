import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { PageHeader } from "@/components/PageHeader";
import { EngagementRibbon } from "@/components/ui/EngagementRibbon";
import { Card, CardHeader, ErrorState, RiskChip, Skeleton, riskColor } from "@/components/ui/primitives";
import { useModelInfo, useOverview } from "@/lib/api/hooks";
import type { RiskBand, StudentSummary } from "@/types";
import { RISK_LABEL } from "@/types";

const BANDS: RiskBand[] = ["High", "Medium", "Low"];

export function DashboardPage() {
  const { data, isLoading, isError, refetch } = useOverview();
  const { data: model } = useModelInfo();

  return (
    <div className="mx-auto max-w-6xl px-4 py-7 sm:px-6 lg:px-10 lg:py-10">
      <PageHeader
        eyebrow="This week"
        title="Who needs your attention"
        lede={
          model
            ? `Across ${data?.total ?? "—"} students, scored at ${Math.round(
                model.checkpointFraction * 100
              )}% of the way through each course — while there is still time to act.`
            : undefined
        }
      />

      {isError ? (
        <Card>
          <ErrorState what="the overview" onRetry={() => refetch()} />
        </Card>
      ) : isLoading || !data ? (
        <OverviewSkeleton />
      ) : (
        <div className="animate-fade-up space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            {BANDS.map((band) => (
              <Card key={band} className="p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-mono text-eyebrow uppercase text-ink-muted">{RISK_LABEL[band]}</p>
                    <p className="nums mt-2 font-display text-4xl font-bold leading-none" style={{ color: riskColor(band) }}>
                      {data.counts[band]}
                    </p>
                    <p className="mt-1.5 text-sm text-ink-muted">
                      {band === "High"
                        ? "reach out this week"
                        : band === "Medium"
                        ? "keep an eye on"
                        : "no action needed"}
                    </p>
                  </div>
                  <span className="h-9 w-1 rounded-full" style={{ background: riskColor(band) }} aria-hidden />
                </div>
              </Card>
            ))}
          </div>

          <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
            <Card>
              <CardHeader
                title="Needs a check-in first"
                eyebrow="Priority"
                action={
                  <Link
                    to="/students?risk=High"
                    className="inline-flex min-h-[44px] items-center gap-1.5 text-sm font-semibold text-brand hover:text-brand-deep"
                  >
                    All students <ArrowRight size={15} aria-hidden />
                  </Link>
                }
              />
              <ul className="divide-y divide-line">
                {data.needsAttention.map((s: StudentSummary) => (
                  <li key={s.id}>
                    <Link
                      to={`/students/${s.id}`}
                      className="flex items-center gap-4 px-5 py-3.5 transition-colors duration-150 hover:bg-brand-soft/50"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-semibold text-ink">{s.name}</p>
                        <p className="mt-0.5 font-mono text-xs text-ink-muted">
                          {s.courseCode} · {s.submittedCount}/{s.expectedCount} submitted · last seen{" "}
                          {s.lastActiveDaysAgo}d ago
                        </p>
                      </div>
                      <div className="hidden w-28 shrink-0 sm:block">
                        <EngagementRibbon activity={s.activity} band={s.riskBand} height={30} />
                      </div>
                      <RiskChip band={s.riskBand} size="sm" />
                    </Link>
                  </li>
                ))}
              </ul>
            </Card>

            <div className="space-y-5">
              <Card>
                <CardHeader title="Spread of the caseload" eyebrow="Distribution" />
                <div className="px-5 py-4">
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={BANDS.map((b) => ({ name: RISK_LABEL[b], value: data.counts[b], band: b }))}
                          dataKey="value"
                          innerRadius="58%"
                          outerRadius="88%"
                          paddingAngle={2}
                          strokeWidth={0}
                        >
                          {BANDS.map((b) => (
                            <Cell key={b} fill={riskColor(b)} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            borderRadius: 10,
                            border: "1px solid #DFE3E1",
                            fontSize: 13,
                            fontFamily: "Public Sans, sans-serif",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <ul className="mt-3 space-y-1.5">
                    {BANDS.map((b) => (
                      <li key={b} className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2 text-ink-muted">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: riskColor(b) }} aria-hidden />
                          {RISK_LABEL[b]}
                        </span>
                        <span className="nums font-mono text-ink">
                          {Math.round((data.counts[b] / data.total) * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </Card>

              <Card>
                <CardHeader title="By course" eyebrow="Breakdown" />
                <ul className="divide-y divide-line">
                  {data.byCourse.map((c: any) => (
                    <li key={c.course.code} className="px-5 py-3">
                      <div className="flex items-baseline justify-between gap-3">
                        <p className="truncate text-sm font-semibold text-ink">{c.course.title}</p>
                        <span className="nums shrink-0 font-mono text-xs text-ink-muted">{c.total}</span>
                      </div>
                      <div className="mt-2 flex h-1.5 gap-0.5 overflow-hidden rounded-full">
                        {(["High", "Medium", "Low"] as RiskBand[]).map((b) => {
                          const n = b === "High" ? c.high : b === "Medium" ? c.medium : c.low;
                          return n ? (
                            <span
                              key={b}
                              style={{ background: riskColor(b), width: `${(n / c.total) * 100}%` }}
                              title={`${n} ${RISK_LABEL[b]}`}
                            />
                          ) : null;
                        })}
                      </div>
                    </li>
                  ))}
                </ul>
              </Card>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function OverviewSkeleton() {
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <Card key={i} className="p-5">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="mt-3 h-9 w-14" />
            <Skeleton className="mt-2 h-3 w-28" />
          </Card>
        ))}
      </div>
      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <Card className="p-5">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-4 py-3">
              <div className="flex-1">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="mt-2 h-3 w-52" />
              </div>
              <Skeleton className="h-6 w-24" />
            </div>
          ))}
        </Card>
        <Card className="p-5">
          <Skeleton className="h-44 w-full" />
        </Card>
      </div>
    </div>
  );
}
