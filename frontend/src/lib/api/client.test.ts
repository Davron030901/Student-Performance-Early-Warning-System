/**
 * Tests for the data layer.
 *
 * The mock client is not throwaway scaffolding: it defines the exact contract
 * the real API must satisfy, and it is what the dashboard is demonstrated on.
 * Filtering, sorting and pagination are real logic with real off-by-one risk.
 */
import { describe, expect, it } from "vitest";
import { fetchCourses, fetchModelInfo, fetchOverview, fetchStudentDetail, fetchStudents } from "./client";
import { COURSES, STUDENTS } from "./mockData";
import type { RiskBand } from "@/types";
import { RISK_LABEL } from "@/types";

const BANDS: RiskBand[] = ["Low", "Medium", "High"];

describe("mock cohort", () => {
  it("is populated enough to look like a real caseload", () => {
    expect(STUDENTS.length).toBeGreaterThanOrEqual(60);
    expect(COURSES.length).toBeGreaterThanOrEqual(3);
  });

  it("gives every student a unique id", () => {
    expect(new Set(STUDENTS.map((s) => s.id)).size).toBe(STUDENTS.length);
  });

  it("spans all three risk bands", () => {
    expect(new Set(STUDENTS.map((s) => s.riskBand))).toEqual(new Set(BANDS));
  });

  it("skews toward students who are doing fine", () => {
    // A roster where most students are in trouble would make triage meaningless.
    const steady = STUDENTS.filter((s) => s.riskBand === "Low").length;
    expect(steady / STUDENTS.length).toBeGreaterThan(0.4);
  });

  it("labels every risk score with the band it falls in", () => {
    // The band is what advisors act on; disagreement with the score would mean
    // the interface shows two different answers at once.
    for (const s of STUDENTS) {
      const expected = s.riskScore <= 0.33 ? "Low" : s.riskScore <= 0.66 ? "Medium" : "High";
      expect(s.riskBand, `${s.id} scored ${s.riskScore}`).toBe(expected);
    }
  });

  it("keeps every score a valid probability", () => {
    for (const s of STUDENTS) {
      expect(s.riskScore).toBeGreaterThanOrEqual(0);
      expect(s.riskScore).toBeLessThanOrEqual(1);
    }
  });

  it("never claims more submissions than were expected", () => {
    for (const s of STUDENTS) {
      expect(s.submittedCount).toBeLessThanOrEqual(s.expectedCount);
      expect(s.submittedCount).toBeGreaterThanOrEqual(0);
    }
  });

  it("assigns every student to a real course", () => {
    const codes = new Set(COURSES.map((c) => c.code));
    for (const s of STUDENTS) expect(codes.has(s.courseCode)).toBe(true);
  });

  it("gives every student activity that crosses the checkpoint", () => {
    // The engagement ribbon depends on having weeks on both sides of the line.
    for (const s of STUDENTS) {
      expect(s.activity.length).toBeGreaterThan(0);
      expect(s.activity.some((a) => a.beforeCheckpoint)).toBe(true);
      expect(s.activity.some((a) => !a.beforeCheckpoint)).toBe(true);
    }
  });

  it("orders activity weeks with the checkpoint as a clean split", () => {
    for (const s of STUDENTS) {
      const before = s.activity.filter((a) => a.beforeCheckpoint);
      const lastBefore = before[before.length - 1].week;
      const firstAfter = s.activity.find((a) => !a.beforeCheckpoint)!.week;
      expect(firstAfter).toBeGreaterThan(lastBefore);
    }
  });

  it("explains every student with plain-language factors", () => {
    for (const s of STUDENTS) {
      expect(s.topFactors.length).toBeGreaterThan(0);
      for (const f of s.topFactors) {
        expect(f.text).toMatch(/\s/);
        expect(f.text[0]).toBe(f.text[0].toUpperCase());
      }
    }
  });

  it("never names a demographic as a reason", () => {
    // Same principle as the backend: a check-in must never be prompted by who
    // a student is.
    const banned = /\b(gender|male|female|region|disability|age|deprivation|ethnic)\b/i;
    for (const s of STUDENTS) {
      for (const f of s.topFactors) expect(f.text, `${s.id}: ${f.text}`).not.toMatch(banned);
    }
  });

  it("explains high-risk students with concerning factors", () => {
    // Explanations must agree with the score, or they undermine it.
    const high = STUDENTS.filter((s) => s.riskBand === "High");
    expect(high.length).toBeGreaterThan(0);
    for (const s of high) {
      expect(s.topFactors.some((f) => f.impact > 0), `${s.id} has no risk-raising factor`).toBe(true);
    }
  });

  it("reports no early score for students who have not submitted", () => {
    for (const s of STUDENTS.filter((x) => x.submittedCount === 0)) {
      expect(s.avgEarlyScore).toBeNull();
    }
  });

  it("is deterministic across imports", async () => {
    const again = (await import("./mockData")).STUDENTS;
    expect(again.map((s) => s.id)).toEqual(STUDENTS.map((s) => s.id));
    expect(again[0].riskScore).toBe(STUDENTS[0].riskScore);
  });
});

describe("fetchStudents", () => {
  it("paginates without dropping or duplicating anyone", async () => {
    const first = await fetchStudents({ page: 1 });
    const second = await fetchStudents({ page: 2 });
    expect(first.students.length).toBe(first.pageSize);
    expect(first.total).toBe(STUDENTS.length);
    const overlap = first.students.filter((s) => second.students.some((o) => o.id === s.id));
    expect(overlap).toHaveLength(0);
  });

  it("returns every student exactly once across all pages", async () => {
    const first = await fetchStudents({ page: 1 });
    const pages = Math.ceil(first.total / first.pageSize);
    const seen: string[] = [];
    for (let p = 1; p <= pages; p++) {
      const page = await fetchStudents({ page: p });
      seen.push(...page.students.map((s) => s.id));
    }
    expect(seen.length).toBe(STUDENTS.length);
    expect(new Set(seen).size).toBe(STUDENTS.length);
  });

  it("returns an empty page past the end rather than erroring", async () => {
    const result = await fetchStudents({ page: 9999 });
    expect(result.students).toHaveLength(0);
    expect(result.total).toBe(STUDENTS.length);
  });

  it("sorts by risk descending by default", async () => {
    const { students } = await fetchStudents({});
    const scores = students.map((s) => s.riskScore);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
  });

  it("sorts by name when asked", async () => {
    const { students } = await fetchStudents({ sortBy: "name" });
    const names = students.map((s) => s.name);
    expect([...names].sort((a, b) => a.localeCompare(b))).toEqual(names);
  });

  it("sorts by longest absence when asked", async () => {
    const { students } = await fetchStudents({ sortBy: "lastActive" });
    const days = students.map((s) => s.lastActiveDaysAgo);
    expect([...days].sort((a, b) => b - a)).toEqual(days);
  });

  it("filters by risk band", async () => {
    for (const band of BANDS) {
      const { students, total } = await fetchStudents({ riskBand: band });
      expect(students.every((s) => s.riskBand === band)).toBe(true);
      expect(total).toBe(STUDENTS.filter((s) => s.riskBand === band).length);
    }
  });

  it("filters by course", async () => {
    const code = COURSES[0].code;
    const { students, total } = await fetchStudents({ course: code });
    expect(students.every((s) => s.courseCode === code)).toBe(true);
    expect(total).toBe(STUDENTS.filter((s) => s.courseCode === code).length);
  });

  it("combines filters rather than letting one override the other", async () => {
    const code = COURSES[0].code;
    const { students } = await fetchStudents({ course: code, riskBand: "Low" });
    expect(students.every((s) => s.courseCode === code && s.riskBand === "Low")).toBe(true);
  });

  it("searches by name, case-insensitively", async () => {
    const target = STUDENTS[3];
    const { students } = await fetchStudents({ search: target.name.toUpperCase() });
    expect(students.some((s) => s.id === target.id)).toBe(true);
  });

  it("searches by student id", async () => {
    const target = STUDENTS[7];
    const { students } = await fetchStudents({ search: target.id });
    expect(students.some((s) => s.id === target.id)).toBe(true);
  });

  it("matches on partial names", async () => {
    const fragment = STUDENTS[2].name.split(" ")[0].slice(0, 4);
    const { students } = await fetchStudents({ search: fragment });
    expect(students.length).toBeGreaterThan(0);
  });

  it("ignores surrounding whitespace in a search", async () => {
    const target = STUDENTS[5];
    const { students } = await fetchStudents({ search: `   ${target.name}   ` });
    expect(students.some((s) => s.id === target.id)).toBe(true);
  });

  it("returns nothing for a search that matches nothing", async () => {
    const { students, total } = await fetchStudents({ search: "zzzz-no-such-student" });
    expect(students).toHaveLength(0);
    expect(total).toBe(0);
  });

  it("treats 'all' as no filter", async () => {
    const filtered = await fetchStudents({ riskBand: "all", course: "all" });
    expect(filtered.total).toBe(STUDENTS.length);
  });
});

describe("fetchStudentDetail", () => {
  it("returns the requested student with the full detail shape", async () => {
    const target = STUDENTS[10];
    const detail = await fetchStudentDetail(target.id);
    expect(detail.id).toBe(target.id);
    expect(detail.topFactors.length).toBeGreaterThan(0);
    expect(detail.checkpointUsed).toBeTruthy();
    expect(detail.modelVersion).toBeTruthy();
    expect(typeof detail.totalClicks).toBe("number");
  });

  it("agrees with the summary shown in the roster", async () => {
    // The detail page must not contradict the row the advisor clicked.
    const { students } = await fetchStudents({});
    const summary = students[0];
    const detail = await fetchStudentDetail(summary.id);
    expect(detail.riskScore).toBe(summary.riskScore);
    expect(detail.riskBand).toBe(summary.riskBand);
    expect(detail.name).toBe(summary.name);
  });

  it("rejects an unknown id", async () => {
    await expect(fetchStudentDetail("does-not-exist")).rejects.toThrow();
  });

  it("only counts pre-checkpoint clicks in the total", async () => {
    // Showing post-checkpoint activity as an input to the score would misstate
    // what the model actually saw.
    const detail = await fetchStudentDetail(STUDENTS[4].id);
    const beforeOnly = detail.activity
      .filter((a) => a.beforeCheckpoint)
      .reduce((sum, a) => sum + a.clicks, 0);
    expect(detail.totalClicks).toBe(beforeOnly);
  });
});

describe("fetchOverview", () => {
  it("counts every student exactly once across the bands", async () => {
    const overview = await fetchOverview();
    const summed = BANDS.reduce((n, b) => n + overview.counts[b], 0);
    expect(summed).toBe(overview.total);
    expect(overview.total).toBe(STUDENTS.length);
  });

  it("puts the highest-risk students at the top of the attention list", async () => {
    const overview = await fetchOverview();
    const scores = overview.needsAttention.map((s: { riskScore: number }) => s.riskScore);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
    expect(scores[0]).toBe(Math.max(...STUDENTS.map((s) => s.riskScore)));
  });

  it("breaks down every course so the bars add up", async () => {
    const overview = await fetchOverview();
    expect(overview.byCourse).toHaveLength(COURSES.length);
    for (const c of overview.byCourse) {
      expect(c.high + c.medium + c.low).toBe(c.total);
    }
  });
});

describe("fetchCourses and fetchModelInfo", () => {
  it("returns courses with a checkpoint inside the course length", async () => {
    const courses = await fetchCourses();
    for (const c of courses) {
      expect(c.checkpointDay).toBeGreaterThan(0);
      expect(c.checkpointDay).toBeLessThan(c.lengthDays);
    }
  });

  it("reports model metrics as valid proportions", async () => {
    const info = await fetchModelInfo();
    expect(info.modelVersion).toBeTruthy();
    expect(info.checkpointFraction).toBeGreaterThan(0);
    expect(info.checkpointFraction).toBeLessThan(1);
    for (const key of ["recall", "precision", "f2", "roc_auc"] as const) {
      expect(info.heldOutMetrics[key]).toBeGreaterThan(0);
      expect(info.heldOutMetrics[key]).toBeLessThanOrEqual(1);
    }
  });
});

describe("advisor-facing language", () => {
  it("never shows raw model vocabulary to the user", () => {
    // "High risk" reads as a verdict about a person; the interface names the
    // action instead.
    expect(RISK_LABEL.High).not.toMatch(/high|risk/i);
    expect(RISK_LABEL.Low).not.toMatch(/low|risk/i);
    expect(RISK_LABEL.Medium).not.toMatch(/medium|risk/i);
  });

  it("labels every band", () => {
    for (const band of BANDS) expect(RISK_LABEL[band]).toBeTruthy();
  });
});
