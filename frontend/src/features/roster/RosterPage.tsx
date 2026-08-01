import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Search, SlidersHorizontal } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EngagementRibbon } from "@/components/ui/EngagementRibbon";
import { Button, Card, EmptyState, ErrorState, RiskChip, Skeleton, cn } from "@/components/ui/primitives";
import { useCourses, useStudents } from "@/lib/api/hooks";
import type { Course, RiskBand } from "@/types";
import { RISK_LABEL } from "@/types";

const BAND_FILTERS: (RiskBand | "all")[] = ["all", "High", "Medium", "Low"];

export function RosterPage() {
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useState("");

  const riskBand = (params.get("risk") as RiskBand | null) ?? "all";
  const course = params.get("course") ?? "all";
  const page = Number(params.get("page") ?? 1);
  const sortBy = (params.get("sort") as "risk" | "name" | "lastActive") ?? "risk";

  const { data: courses } = useCourses();
  const { data, isLoading, isError, refetch, isPlaceholderData } = useStudents({
    riskBand,
    course,
    search,
    page,
    sortBy,
  });

  const update = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value === "all") next.delete(key);
    else next.set(key, value);
    if (key !== "page") next.delete("page");
    setParams(next);
  };

  const clearAll = () => {
    setSearch("");
    setParams(new URLSearchParams());
  };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.pageSize)) : 1;

  return (
    <div className="mx-auto max-w-6xl px-4 py-7 sm:px-6 lg:px-10 lg:py-10">
      <PageHeader
        eyebrow="Roster"
        title="All students"
        lede="Sorted by how much attention they need. Every score comes from activity recorded before the course checkpoint."
      />

      {/* Filters */}
      <div className="mb-5 space-y-3">
        <div className="relative">
          <Search size={17} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-muted" aria-hidden />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or student ID"
            aria-label="Search students by name or ID"
            className="min-h-[46px] w-full rounded-chip border border-line bg-surface pl-11 pr-4 text-[15px] text-ink placeholder:text-ink-muted/70"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by attention level">
            {BAND_FILTERS.map((b) => (
              <button
                key={b}
                onClick={() => update("risk", b)}
                aria-pressed={riskBand === b}
                className={cn(
                  "min-h-[38px] rounded-chip border px-3 text-sm font-semibold transition-colors duration-150",
                  riskBand === b
                    ? "border-ink bg-ink text-white"
                    : "border-line bg-surface text-ink-muted hover:border-ink-muted hover:text-ink"
                )}
              >
                {b === "all" ? "Everyone" : RISK_LABEL[b]}
              </button>
            ))}
          </div>

          <div className="flex w-full items-center gap-2 sm:ml-auto sm:w-auto">
            <label className="sr-only" htmlFor="course-filter">
              Filter by course
            </label>
            <select
              id="course-filter"
              value={course}
              onChange={(e) => update("course", e.target.value)}
              className="min-h-[38px] min-w-0 flex-1 rounded-chip border border-line bg-surface px-3 text-sm font-medium text-ink sm:flex-none"
            >
              <option value="all">All courses</option>
              {courses?.map((c: Course) => (
                <option key={c.code} value={c.code}>
                  {c.code} — {c.title}
                </option>
              ))}
            </select>

            <label className="sr-only" htmlFor="sort-by">
              Sort students
            </label>
            <select
              id="sort-by"
              value={sortBy}
              onChange={(e) => update("sort", e.target.value)}
              className="min-h-[38px] min-w-0 flex-1 rounded-chip border border-line bg-surface px-3 text-sm font-medium text-ink sm:flex-none"
            >
              <option value="risk">Most attention first</option>
              <option value="name">Name A–Z</option>
              <option value="lastActive">Longest since sign-in</option>
            </select>
          </div>
        </div>
      </div>

      <Card className={cn("overflow-hidden transition-opacity", isPlaceholderData && "opacity-60")}>
        {isError ? (
          <ErrorState what="the student list" onRetry={() => refetch()} />
        ) : isLoading ? (
          <RosterSkeleton />
        ) : !data || data.students.length === 0 ? (
          <EmptyState
            title="No students match these filters"
            hint="Try widening the attention level, choosing a different course, or clearing the search."
            action={
              <Button variant="ghost" onClick={clearAll}>
                <SlidersHorizontal size={15} aria-hidden /> Clear filters
              </Button>
            }
          />
        ) : (
          <>
            {/* Desktop: table. Mobile: the same rows restructured as cards — never a sideways-scrolling table. */}
            <div className="hidden lg:block">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-line text-left">
                    {["Student", "Course", "Submitted", "Last seen", "Engagement to checkpoint", "Attention"].map((h) => (
                      <th key={h} className="px-5 py-3 font-mono text-eyebrow uppercase text-ink-muted">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.students.map((s) => (
                    <tr key={s.id} className="group transition-colors duration-150 hover:bg-brand-soft/40">
                      <td className="px-5 py-3">
                        <Link to={`/students/${s.id}`} className="font-semibold text-ink hover:text-brand">
                          {s.name}
                        </Link>
                        <p className="nums font-mono text-xs text-ink-muted">{s.id}</p>
                      </td>
                      <td className="px-5 py-3 font-mono text-sm text-ink-muted">{s.courseCode}</td>
                      <td className="nums px-5 py-3 font-mono text-sm text-ink">
                        {s.submittedCount}/{s.expectedCount}
                      </td>
                      <td className="nums px-5 py-3 font-mono text-sm text-ink-muted">{s.lastActiveDaysAgo}d ago</td>
                      <td className="w-40 px-5 py-3">
                        <EngagementRibbon activity={s.activity} band={s.riskBand} height={28} />
                      </td>
                      <td className="px-5 py-3">
                        <RiskChip band={s.riskBand} size="sm" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <ul className="divide-y divide-line lg:hidden">
              {data.students.map((s) => (
                <li key={s.id}>
                  <Link to={`/students/${s.id}`} className="block px-4 py-4 transition-colors duration-150 active:bg-brand-soft/60">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate font-semibold text-ink">{s.name}</p>
                        <p className="nums mt-0.5 font-mono text-xs text-ink-muted">
                          {s.id} · {s.courseCode}
                        </p>
                      </div>
                      <RiskChip band={s.riskBand} size="sm" />
                    </div>
                    <div className="mt-3">
                      <EngagementRibbon activity={s.activity} band={s.riskBand} height={34} />
                    </div>
                    <p className="nums mt-2 font-mono text-xs text-ink-muted">
                      {s.submittedCount}/{s.expectedCount} submitted · last seen {s.lastActiveDaysAgo}d ago
                    </p>
                  </Link>
                </li>
              ))}
            </ul>

            <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3 sm:px-5">
              <p className="nums font-mono text-xs text-ink-muted">
                {(data.page - 1) * data.pageSize + 1}–{Math.min(data.page * data.pageSize, data.total)} of {data.total}
              </p>
              <div className="flex gap-2">
                <Button variant="ghost" disabled={page <= 1} onClick={() => update("page", String(page - 1))}>
                  Previous
                </Button>
                <Button variant="ghost" disabled={page >= totalPages} onClick={() => update("page", String(page + 1))}>
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

function RosterSkeleton() {
  return (
    <div className="divide-y divide-line">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 px-5 py-4">
          <div className="flex-1">
            <Skeleton className="h-4 w-44" />
            <Skeleton className="mt-2 h-3 w-24" />
          </div>
          <Skeleton className="hidden h-7 w-36 lg:block" />
          <Skeleton className="h-6 w-28" />
        </div>
      ))}
    </div>
  );
}
