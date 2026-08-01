/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Core surface palette ─────────────────────────────────────────
        // Cool, institutional, calm. Deliberately not warm-cream (a current
        // AI-design default) and not near-black.
        ink: { DEFAULT: "#16232E", soft: "#22323F", muted: "#5E6E75" },
        paper: "#F4F6F5",
        surface: "#FFFFFF",
        line: "#DFE3E1",

        // ── Brand ────────────────────────────────────────────────────────
        // Deep verdigris. Chosen over the default SaaS indigo/violet because
        // this is a care-and-support tool, not a growth dashboard: a settled
        // green-teal reads as steady and institutional rather than urgent.
        brand: { DEFAULT: "#17706A", deep: "#0F5450", soft: "#E6F0EE" },

        // ── Risk scale ───────────────────────────────────────────────────
        // Blue → ochre → clay. Avoids the red/green axis that dichromatic
        // viewers struggle with; every use is also paired with an icon and a
        // text label, so colour is never the sole carrier of meaning.
        risk: {
          low: "#3D6E8F",
          lowSoft: "#E8EFF4",
          medium: "#B07D2B",
          mediumSoft: "#F7EEDC",
          high: "#A8443A",
          highSoft: "#F6E7E5",
        },
      },
      fontFamily: {
        // Display: variable grotesque with genuinely idiosyncratic widths —
        // gives the product a face without reaching for a high-contrast serif.
        display: ['"Bricolage Grotesque"', "system-ui", "sans-serif"],
        // Body: designed for public-sector interfaces; legible at small sizes,
        // which matters in a data-dense roster.
        sans: ['"Public Sans"', "system-ui", "sans-serif"],
        // Utility: numerals, IDs, week markers — the "instrument readout" voice.
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      fontSize: {
        "display-lg": ["2.75rem", { lineHeight: "1.05", letterSpacing: "-0.03em" }],
        "display-md": ["1.875rem", { lineHeight: "1.15", letterSpacing: "-0.02em" }],
        eyebrow: ["0.6875rem", { lineHeight: "1", letterSpacing: "0.12em" }],
      },
      borderRadius: { card: "14px", chip: "8px" },
      boxShadow: {
        card: "0 1px 2px rgba(22,35,46,0.04), 0 4px 16px rgba(22,35,46,0.05)",
        lift: "0 2px 6px rgba(22,35,46,0.07), 0 12px 28px rgba(22,35,46,0.09)",
      },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { "fade-up": "fade-up 260ms cubic-bezier(0.22,1,0.36,1) both" },
    },
  },
  plugins: [],
};
