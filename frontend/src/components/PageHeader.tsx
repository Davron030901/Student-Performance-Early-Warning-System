export function PageHeader({ eyebrow, title, lede }: { eyebrow: string; title: string; lede?: string }) {
  return (
    <div className="mb-6">
      <p className="mb-2 font-mono text-eyebrow uppercase text-brand">{eyebrow}</p>
      <h1 className="font-display text-display-md font-bold text-ink sm:text-display-lg">{title}</h1>
      {lede && <p className="mt-2.5 max-w-2xl text-[15px] leading-relaxed text-ink-muted">{lede}</p>}
    </div>
  );
}
