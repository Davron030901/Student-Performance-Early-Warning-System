# Course Signals — Advisor Dashboard (Frontend)

The human-facing side of the EDU-02 early-warning system. Academic advisors use
it to see which students may need support early in a term, understand why, and
act.

Backend: [`../backend`](../backend)

---

## Quickstart

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

It runs immediately on seeded mock data — no backend required, and nothing to
configure to see a fully populated dashboard.

```bash
npm run build        # type-check + production build
npm run preview      # serve the build
npm run lint         # tsc --noEmit
```

## Screens

**Overview** — the caseload at a glance: counts by attention level, the five
students who need a check-in first, distribution, and a per-course breakdown.

**Roster** — all 72 students, searchable by name or ID, filterable by attention
level and course, sortable. Filters live in the URL, so a filtered view can be
bookmarked or shared. A real table on desktop; stacked cards below 1024px.

**Student detail** — the score and what drove it, in plain language, with the
full-size engagement ribbon and the underlying figures recorded at the checkpoint.

Loading, empty and error states are designed rather than defaulted: skeletons
that mirror the real layout, an empty state that suggests what to change, and an
error state that explains what happened and keeps the advisor's filters intact.

## Connecting the real backend

All server access lives in `src/lib/api/client.ts`. No component imports anything
else, so switching is a config change:

```bash
cp .env.example .env
# VITE_API_BASE_URL=http://localhost:8000
# VITE_USE_MOCK=false
```

The client already maps the backend's snake_case (`model_version`,
`checkpoint_fraction`) to the camelCase the UI uses. The backend enables CORS for
`localhost:5173`.

Two endpoints the dashboard wants are not in the backend yet —
`GET /api/v1/students` and `GET /api/v1/students/{id}` — because the backend
scores a supplied feature payload rather than storing a cohort. Adding a thin
roster store, or having the frontend batch through `POST /api/v1/predict/batch`,
closes the gap. `GET /api/v1/model/info` is live and already wired.

## Mock data

`src/lib/api/mockData.ts` seeds 72 students across 4 courses from a fixed PRNG,
so every reload is identical. The generation is causal rather than random: a
latent engagement level drives weekly activity, submission behaviour and scores,
and the risk score is derived from those same signals — so the explanations shown
on the detail page genuinely line up with the student's numbers, which random
data would not.

The spread is skewed toward students who are doing fine (roughly 61% Steady),
because a roster where most students are in trouble would make triage meaningless.

To exercise the error state, run `window.__forceApiError = true` in the console
and reload.

## Stack

React 18 · TypeScript · Vite 6 · Tailwind CSS 3 · TanStack Query · React Router 6
· Recharts · lucide-react

UI primitives (`src/components/ui/primitives.tsx`) are hand-rolled in the
shadcn/ui spirit rather than pulled in via its CLI — the set needed here is
small, and it keeps the dependency surface and the design tokens under direct
control.

## Layout

```
src/
├── components/
│   ├── ui/primitives.tsx        # Card, Button, RiskChip, Skeleton, Empty, Error
│   ├── ui/EngagementRibbon.tsx  # ← the signature element
│   ├── AppShell.tsx             # sidebar → bottom-nav responsive shell
│   └── PageHeader.tsx
├── features/
│   ├── dashboard/               # overview
│   ├── roster/                  # searchable, filterable list
│   └── student-detail/          # score, factors, ribbon, figures
├── lib/api/
│   ├── client.ts                # ← the single swappable data layer
│   ├── mockData.ts              # seeded demo cohort
│   └── hooks.ts                 # TanStack Query wrappers
└── types/
```

## Design

Palette, type pairing, the ribbon rationale and the language choices are
documented in [DESIGN.md](DESIGN.md).

## Verified

Rendering was checked in a headless browser rather than assumed:

- All three screens render with correct data; the detail page matches the row
  clicked to reach it.
- Filtering, search, sorting, pagination and URL state work.
- Empty and error states appear under the right conditions, with a working retry.
- No horizontal overflow at 375px, 768px or 1440px.
- The desktop table is replaced by cards below 1024px; bottom-nav targets are 58px.
- Keyboard tabbing produces a visible focus ring.

> The Google Fonts stylesheet is blocked in some sandboxed networks; the app falls
> back to `system-ui` and remains fully usable. In a normal environment the three
> faces load as intended.
