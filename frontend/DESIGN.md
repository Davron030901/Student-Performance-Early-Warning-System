# Design system

## The brief, restated

A tool advisors open every week to answer one question: **who needs my attention,
and why?** It handles real students' academic futures, so it has to feel
trustworthy and calm — closer to a good clinical instrument than a growth
dashboard — while still being pleasant enough to live in daily.

## Calibration: what this deliberately isn't

Three looks currently dominate AI-generated interface design, and all three were
ruled out at the start because they would have been defaults rather than choices:

1. Warm cream ground (~#F4F1EA), high-contrast serif, terracotta accent (~#D97757).
2. Near-black ground with a single acid-green or vermilion accent.
3. Broadsheet layout — hairline rules, zero radius, dense newspaper columns.

The second is far too alarming for a tool about struggling students. The third
turns a triage list into something that reads like an audit. The first is simply
everywhere. What follows was chosen against this brief instead.

## Colour

| Token | Hex | Role |
|---|---|---|
| `ink` | `#16232E` | Sidebar, headings — a deep blue-slate, cool and institutional |
| `paper` | `#F4F6F5` | App ground; very slightly *cool*, to stay away from cream |
| `surface` | `#FFFFFF` | Cards |
| `line` | `#DFE3E1` | Borders and dividers |
| `brand` | `#17706A` | Deep verdigris — primary actions, active nav |
| `risk.low` | `#3D6E8F` | Steady |
| `risk.medium` | `#B07D2B` | Worth a look |
| `risk.high` | `#A8443A` | Needs a check-in |

**Why verdigris rather than the default SaaS indigo.** Indigo/violet is the
reflexive choice for a product dashboard and carries a growth-metrics
connotation that is wrong here. A settled green-teal reads as steady and
institutional — the colour of a thing that looks after something — without the
urgency of red or the coldness of pure blue.

**Why the risk scale avoids red/amber/green.** The conventional traffic-light
scale fails exactly the people it needs to serve: red/green is the axis
dichromatic viewers struggle with most. This scale runs **cool blue → ochre →
clay red**, which stays separable under deuteranopia and protanopia. On top of
that, colour is never the sole carrier of meaning — every risk chip pairs its hue
with a **distinct icon shape** (check / eye / alert) *and* an explicit text
label. Remove all colour from the interface and it still reads correctly.

The clay red is deliberately muted rather than a fire-engine red. A student who
needs a conversation is not an emergency, and the interface should not shout.

## Type

| Role | Face | Why |
|---|---|---|
| Display | **Bricolage Grotesque** | A variable grotesque with genuinely idiosyncratic widths. Gives the product a face without reaching for the high-contrast serif that every AI-designed page currently uses. |
| Body | **Public Sans** | Drawn for public-sector interfaces. Legible at small sizes, which is the actual constraint in a data-dense roster, and fitting for an institution. |
| Utility | **IBM Plex Mono** | Numerals, student IDs, week markers, eyebrows. Gives figures an instrument-readout voice and keeps columns aligned. |

Tabular numerals (`.nums`) are applied wherever figures are compared down a
column, so digits don't shift between rows.

## The signature element: the engagement ribbon

Everything else here is, honestly, a competent dashboard. This is the piece that
carries the idea.

A student's weekly course-site activity is drawn as a small area ribbon, cut by a
**hard dashed vertical rule at the prediction checkpoint**. Everything left of
the rule is what the model could see. Everything right of it is greyed and
hatched — time that has genuinely passed for the student, but that the prediction
knows nothing about.

That distinction *is* the product. An early-warning system is defined by the fact
that it commits to a judgement on partial information, and the most common way
these tools mislead people is by hiding that. So the ribbon appears at every
scale — 28px tall in a roster row, 30px on the priority list, 150px on the detail
page — and it is the same component each time. An advisor cannot look at a score
in this interface without also seeing how much of the term it was allowed to know
about.

It also does ordinary work well: the shape of the line tells you at a glance
whether a student is climbing, holding, or sliding, which is information no
single number conveys.

## Layout

Dark ink sidebar on desktop (≥1024px), collapsing to a light top bar plus a
two-item bottom navigation on smaller screens. Content sits on `paper` in cards
with a 14px radius and a soft two-layer shadow. Generous whitespace; one idea per
card.

The roster is the layout decision that matters most: on desktop it is a real
table, because scanning many students down aligned columns is the job. Below
1024px it does **not** become a horizontally-scrolling table — it restructures
into stacked cards, each carrying the same information in reading order, with the
ribbon given more height because there is room for it.

## Motion

Deliberate and quiet. Page content rises 6px on entry (260ms, custom easing).
Transitions are 150ms. Loading uses **skeletons that mirror the real layout**
rather than spinners, so the page doesn't jump when data lands. All of it is
disabled under `prefers-reduced-motion`.

## Words

Words are design material here, and the risk vocabulary is the clearest example.
The API returns `Low` / `Medium` / `High`. The interface never shows those:

| API | Interface |
|---|---|
| Low | **Steady** |
| Medium | **Worth a look** |
| High | **Needs a check-in** |

"High risk" sounds like a verdict about a person. It isn't — it's a probability
attached to a moment, from a model that misses roughly one struggling student in
five. "Needs a check-in" says the same thing while naming the action, which is
also the only thing the advisor can actually do with it.

The same principle runs through the rest: the empty state suggests what to
change, the error state explains what happened and preserves the advisor's
filters, and the detail page states plainly that demographic factors are excluded
from explanations so that a check-in is never prompted by who a student is.

## Accessibility floor

Built in, not bolted on: WCAG AA contrast throughout, full keyboard navigation
with a visible 2px brand focus ring, ≥44px touch targets on mobile, semantic
landmarks and headings, `aria-label` on every icon-only control, `role="img"`
with descriptive labels on the ribbons, `aria-pressed` on filter toggles, and
`sr-only` labels on the selects.

Verified at 375px, 768px and 1440px with no horizontal overflow.
