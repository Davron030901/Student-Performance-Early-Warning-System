import { NavLink, Outlet } from "react-router-dom";
import { GraduationCap, LayoutDashboard, Users } from "lucide-react";
import { cn } from "./ui/primitives";
import { useModelInfo } from "@/lib/api/hooks";

const NAV = [
  { to: "/", label: "Overview", Icon: LayoutDashboard, end: true },
  { to: "/students", label: "Students", Icon: Users, end: false },
];

export function AppShell() {
  const { data: model } = useModelInfo();

  return (
    <div className="min-h-dvh lg:flex">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 flex-col justify-between bg-ink px-5 py-6 lg:flex">
        <div>
          <div className="mb-9 flex items-center gap-2.5">
            <span className="grid h-9 w-9 place-items-center rounded-chip bg-brand">
              <GraduationCap size={19} className="text-white" strokeWidth={2.2} aria-hidden />
            </span>
            <span className="font-display text-[15px] font-bold leading-tight tracking-tight text-white">
              Course
              <br />
              Signals
            </span>
          </div>

          <nav className="flex flex-col gap-1" aria-label="Main">
            {NAV.map(({ to, label, Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "flex min-h-[44px] items-center gap-3 rounded-chip px-3 text-sm font-semibold transition-colors duration-150",
                    isActive ? "bg-white/10 text-white" : "text-white/60 hover:bg-white/5 hover:text-white"
                  )
                }
              >
                <Icon size={17} strokeWidth={2.1} aria-hidden />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>

        {model && (
          <div className="rounded-card border border-white/10 bg-white/5 p-3.5">
            <p className="font-mono text-eyebrow uppercase text-white/45">Model</p>
            <p className="mt-1.5 font-mono text-xs text-white/80">{model.modelVersion}</p>
            <p className="mt-2 text-xs leading-relaxed text-white/50">
              Predicts at {Math.round(model.checkpointFraction * 100)}% of the course. Catches about{" "}
              {Math.round(model.heldOutMetrics.recall * 100)} in 100 students who go on to struggle.
            </p>
          </div>
        )}
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-20 flex items-center gap-2.5 border-b border-line bg-surface/95 px-4 py-3 backdrop-blur lg:hidden">
        <span className="grid h-8 w-8 place-items-center rounded-chip bg-brand">
          <GraduationCap size={17} className="text-white" strokeWidth={2.2} aria-hidden />
        </span>
        <span className="font-display text-sm font-bold tracking-tight text-ink">Course Signals</span>
      </header>

      <main className="flex-1 pb-24 lg:pb-0">
        <Outlet />
      </main>

      {/* Mobile bottom navigation — 44px+ touch targets */}
      <nav
        className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-2 border-t border-line bg-surface/95 backdrop-blur lg:hidden"
        aria-label="Main"
      >
        {NAV.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "flex min-h-[58px] flex-col items-center justify-center gap-1 text-xs font-semibold transition-colors duration-150",
                isActive ? "text-brand" : "text-ink-muted"
              )
            }
          >
            <Icon size={19} strokeWidth={2.1} aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
