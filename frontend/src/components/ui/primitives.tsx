import type { ReactNode } from "react";
import { AlertCircle, CheckCircle2, Eye, RefreshCw } from "lucide-react";
import type { RiskBand } from "@/types";
import { RISK_LABEL } from "@/types";

export function cn(...parts: (string | false | undefined | null)[]) {
  return parts.filter(Boolean).join(" ");
}

/* ── Card ─────────────────────────────────────────────────────────────── */
export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-card border border-line bg-surface shadow-card", className)}>{children}</div>
  );
}

export function CardHeader({ title, eyebrow, action }: { title: string; eyebrow?: string; action?: ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4 border-b border-line px-5 py-4">
      <div>
        {eyebrow && (
          <p className="mb-1 font-mono text-eyebrow uppercase text-ink-muted">{eyebrow}</p>
        )}
        <h2 className="font-display text-lg font-semibold tracking-tight text-ink">{title}</h2>
      </div>
      {action}
    </div>
  );
}

/* ── Risk chip ────────────────────────────────────────────────────────── */
const RISK_STYLES: Record<RiskBand, { chip: string; dot: string; Icon: typeof CheckCircle2 }> = {
  Low: { chip: "bg-risk-lowSoft text-risk-low", dot: "bg-risk-low", Icon: CheckCircle2 },
  Medium: { chip: "bg-risk-mediumSoft text-risk-medium", dot: "bg-risk-medium", Icon: Eye },
  High: { chip: "bg-risk-highSoft text-risk-high", dot: "bg-risk-high", Icon: AlertCircle },
};

/**
 * Colour is never the only carrier of meaning here: every chip pairs a hue
 * with a distinct icon shape and an explicit text label.
 */
export function RiskChip({ band, size = "md" }: { band: RiskBand; size?: "sm" | "md" }) {
  const { chip, Icon } = RISK_STYLES[band];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-chip font-semibold",
        chip,
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm"
      )}
    >
      <Icon size={size === "sm" ? 13 : 15} strokeWidth={2.4} aria-hidden />
      {RISK_LABEL[band]}
    </span>
  );
}

export function riskColor(band: RiskBand) {
  return { Low: "#3D6E8F", Medium: "#B07D2B", High: "#A8443A" }[band];
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
export function Button({
  children,
  onClick,
  variant = "primary",
  type = "button",
  className,
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "quiet";
  type?: "button" | "submit";
  className?: string;
  disabled?: boolean;
}) {
  const styles = {
    primary: "bg-brand text-white hover:bg-brand-deep",
    ghost: "border border-line bg-surface text-ink hover:border-ink-muted",
    quiet: "text-ink-muted hover:text-ink hover:bg-brand-soft",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "inline-flex min-h-[44px] items-center justify-center gap-2 rounded-chip px-4 text-sm font-semibold transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45",
        styles,
        className
      )}
    >
      {children}
    </button>
  );
}

/* ── Loading / empty / error ──────────────────────────────────────────── */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-chip bg-line/60", className)}>
      <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/70 to-transparent motion-safe:animate-[shimmer_1.6s_infinite]" />
    </div>
  );
}

export function EmptyState({ title, hint, action }: { title: string; hint: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-14 text-center">
      <h3 className="font-display text-base font-semibold text-ink">{title}</h3>
      <p className="max-w-sm text-sm text-ink-muted">{hint}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function ErrorState({ onRetry, what = "this view" }: { onRetry: () => void; what?: string }) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-14 text-center">
      <AlertCircle className="text-risk-high" size={26} strokeWidth={2} aria-hidden />
      <div>
        <h3 className="font-display text-base font-semibold text-ink">Couldn't load {what}</h3>
        <p className="mt-1 max-w-sm text-sm text-ink-muted">
          The prediction service didn't respond. Your filters are still set — retrying will keep them.
        </p>
        <p className="mt-2 max-w-sm text-xs text-ink-muted/80">
          On a free hosting plan the service sleeps when unused and can take up to a minute to wake.
          If this is the first visit in a while, waiting a moment and retrying usually works.
        </p>
      </div>
      <Button variant="ghost" onClick={onRetry}>
        <RefreshCw size={15} aria-hidden /> Try again
      </Button>
    </div>
  );
}
