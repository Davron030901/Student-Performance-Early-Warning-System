import type { Course, RiskBand, StudentDetail, WeeklyActivity } from "@/types";

export const COURSES: Course[] = [
  { code: "AAA", title: "Foundations of Social Policy", presentation: "2026J", lengthDays: 268, checkpointDay: 80 },
  { code: "BBB", title: "Introductory Statistics", presentation: "2026J", lengthDays: 234, checkpointDay: 70 },
  { code: "CCC", title: "Computing & IT Practice", presentation: "2026J", lengthDays: 269, checkpointDay: 81 },
  { code: "DDD", title: "Environmental Science", presentation: "2026B", lengthDays: 240, checkpointDay: 72 },
];

const FIRST_NAMES = [
  "Amara", "Yusuf", "Priya", "Tomas", "Nadia", "Ewan", "Chloe", "Ravi", "Marta", "Idris",
  "Freya", "Kwame", "Sinead", "Omar", "Lena", "Hugo", "Aisha", "Callum", "Mei", "Sofia",
  "Dario", "Halima", "Jonas", "Zainab", "Rory", "Tess", "Bilal", "Anika", "Felix", "Noor",
  "Kiera", "Milos", "Sana", "Declan", "Yara", "Otto", "Nia", "Emil", "Layla", "Gus",
];
const LAST_NAMES = [
  "Okafor", "Demir", "Raman", "Novak", "Haddad", "Fraser", "Bennett", "Iyer", "Kowalski", "Abubakar",
  "Lindqvist", "Mensah", "O'Rourke", "Farouk", "Vogel", "Marchetti", "Bello", "Reid", "Chen", "Duarte",
  "Ferro", "Osman", "Larsen", "Karim", "Mackay", "Whitfield", "Aziz", "Sharma", "Brandt", "Haq",
];

/** Deterministic PRNG so the demo data is identical on every load. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(20260729);

function bandFor(score: number): RiskBand {
  if (score <= 0.33) return "Low";
  if (score <= 0.66) return "Medium";
  return "High";
}

function buildActivity(engagement: number, weeksBefore: number, totalWeeks: number, r: () => number): WeeklyActivity[] {
  const weeks: WeeklyActivity[] = [];
  // Disengaging students trend downward; steady students hold or climb.
  const drift = engagement < 0.4 ? -0.13 : engagement > 0.7 ? 0.05 : -0.02;
  let level = 18 + engagement * 90;
  for (let w = 1; w <= totalWeeks; w++) {
    level = Math.max(0, level * (1 + drift) + (r() - 0.5) * 18);
    weeks.push({
      week: w,
      clicks: Math.round(Math.max(0, level)),
      beforeCheckpoint: w <= weeksBefore,
    });
  }
  return weeks;
}

function factorsFor(engagement: number, submitted: number, expected: number, score: number, lastActive: number) {
  const factors: { text: string; impact: number }[] = [];
  if (submitted === 0) factors.push({ text: "No assessment submitted yet this term", impact: 0.34 });
  else if (submitted < expected) factors.push({ text: `Missed ${expected - submitted} of ${expected} early assessments`, impact: 0.21 });
  else factors.push({ text: "All early assessments submitted", impact: -0.19 });

  if (engagement < 0.35) factors.push({ text: "Course site activity well below the module average", impact: 0.28 });
  else if (engagement > 0.7) factors.push({ text: "Consistently active on the course site", impact: -0.22 });
  else factors.push({ text: "Course site activity is around the module average", impact: -0.04 });

  if (lastActive > 10) factors.push({ text: `No sign-in for ${lastActive} days`, impact: 0.24 });
  else if (lastActive <= 2) factors.push({ text: "Signed in within the last two days", impact: -0.12 });

  if (engagement < 0.5) factors.push({ text: "Engagement has fallen since the first fortnight", impact: 0.17 });
  else factors.push({ text: "Engagement has held steady since the start", impact: -0.09 });

  if (score < 0.5 && submitted > 0) factors.push({ text: "Early assessment scores below the pass threshold", impact: 0.2 });

  return factors.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 4);
}

function buildStudent(index: number): StudentDetail {
  const r = rand;
  const course = COURSES[index % COURSES.length];
  const first = FIRST_NAMES[Math.floor(r() * FIRST_NAMES.length)];
  const last = LAST_NAMES[Math.floor(r() * LAST_NAMES.length)];

  // Skewed toward doing fine — a roster where most students are in trouble
  // would be unrealistic and would make the triage view meaningless.
  const roll = r();
  const engagement = roll < 0.58 ? 0.62 + r() * 0.38 : roll < 0.82 ? 0.35 + r() * 0.3 : r() * 0.35;

  const expected = 2 + (index % 2);
  const submitted = engagement > 0.6 ? expected : engagement > 0.35 ? Math.max(0, expected - 1) : 0;
  const avgScore = submitted === 0 ? null : Math.round(38 + engagement * 52 + (r() - 0.5) * 12);
  const lastActive = Math.round((1 - engagement) * 26 + r() * 3);

  // Risk is driven by the same signals the model uses, so explanations line up.
  const raw =
    0.44 * (1 - engagement) +
    0.28 * (1 - submitted / expected) +
    0.16 * Math.min(1, lastActive / 20) +
    0.12 * (avgScore === null ? 1 : 1 - avgScore / 100);
  const riskScore = Math.min(0.985, Math.max(0.012, raw + (r() - 0.5) * 0.08));

  const totalWeeks = Math.round(course.lengthDays / 7);
  const weeksBefore = Math.round(course.checkpointDay / 7);
  const activity = buildActivity(engagement, weeksBefore, Math.min(totalWeeks, weeksBefore + 6), r);

  return {
    id: `S-${(10240 + index * 7).toString()}`,
    name: `${first} ${last}`,
    courseCode: course.code,
    riskScore: Number(riskScore.toFixed(3)),
    riskBand: bandFor(riskScore),
    submittedCount: submitted,
    expectedCount: expected,
    lastActiveDaysAgo: lastActive,
    activity,
    topFactors: factorsFor(engagement, submitted, expected, avgScore === null ? 0 : avgScore / 100, lastActive),
    avgEarlyScore: avgScore,
    onTimeRate: submitted === 0 ? 0 : Math.min(1, 0.45 + engagement * 0.55),
    totalClicks: activity.filter((a) => a.beforeCheckpoint).reduce((s, a) => s + a.clicks, 0),
    activeDays: Math.round(engagement * 46 + r() * 6),
    registeredDay: Math.round(-24 + (1 - engagement) * 30),
    checkpointUsed: "30% of course length",
    modelVersion: "xgb-v1.0",
  };
}

export const STUDENTS: StudentDetail[] = Array.from({ length: 72 }, (_, i) => buildStudent(i));

export const MODEL_INFO = {
  modelVersion: "xgb-v1.0",
  trainedAt: "2026-07-29 03:41:56 UTC",
  checkpointFraction: 0.3,
  nTrainingRows: 2240,
  heldOutMetrics: { recall: 0.7812, precision: 0.7143, f2: 0.7669, roc_auc: 0.9106 },
};
