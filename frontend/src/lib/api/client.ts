/**
 * The single data-access layer for the whole app.
 *
 * Everything the UI knows about the server lives behind these functions. To
 * point the dashboard at the real FastAPI backend, set VITE_API_BASE_URL in
 * .env and flip USE_MOCK to false — no component changes are required, because
 * no component imports anything but this module.
 */
import type { ModelInfo, StudentDetail, StudentListResponse, StudentQuery } from "@/types";
import { COURSES, MODEL_INFO, STUDENTS } from "./mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

const PAGE_SIZE = 12;

/** Simulated latency, so loading states are visible during development. */
const delay = (ms = 420) => new Promise((res) => setTimeout(res, ms));

/** Set to true in the browser console to exercise the error state: window.__forceApiError = true */
declare global {
  interface Window {
    __forceApiError?: boolean;
  }
}

function assertNotForcedError() {
  if (typeof window !== "undefined" && window.__forceApiError) {
    throw new Error("Simulated network failure");
  }
}

export async function fetchCourses() {
  if (USE_MOCK) {
    await delay(150);
    assertNotForcedError();
    return COURSES;
  }
  const r = await fetch(`${API_BASE}/api/v1/courses`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
}

export async function fetchStudents(query: StudentQuery = {}): Promise<StudentListResponse> {
  if (USE_MOCK) {
    await delay();
    assertNotForcedError();

    const { course, riskBand = "all", search = "", page = 1, sortBy = "risk" } = query;
    let rows = [...STUDENTS];

    if (course && course !== "all") rows = rows.filter((s) => s.courseCode === course);
    if (riskBand !== "all") rows = rows.filter((s) => s.riskBand === riskBand);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((s) => s.name.toLowerCase().includes(q) || s.id.toLowerCase().includes(q));
    }

    rows.sort((a, b) => {
      if (sortBy === "name") return a.name.localeCompare(b.name);
      if (sortBy === "lastActive") return b.lastActiveDaysAgo - a.lastActiveDaysAgo;
      return b.riskScore - a.riskScore;
    });

    const total = rows.length;
    const start = (page - 1) * PAGE_SIZE;
    return { students: rows.slice(start, start + PAGE_SIZE), total, page, pageSize: PAGE_SIZE };
  }

  const params = new URLSearchParams();
  if (query.course && query.course !== "all") params.set("course", query.course);
  if (query.riskBand && query.riskBand !== "all") params.set("riskBand", query.riskBand);
  if (query.search) params.set("search", query.search);
  params.set("page", String(query.page ?? 1));

  const r = await fetch(`${API_BASE}/api/v1/students?${params}`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
}

export async function fetchStudentDetail(id: string): Promise<StudentDetail> {
  if (USE_MOCK) {
    await delay(320);
    assertNotForcedError();
    const found = STUDENTS.find((s) => s.id === id);
    if (!found) throw new Error(`No student with id ${id}`);
    return found;
  }
  const r = await fetch(`${API_BASE}/api/v1/students/${id}`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
}

/** Caseload-wide counts, used by the overview. */
export async function fetchOverview() {
  if (USE_MOCK) {
    await delay(300);
    assertNotForcedError();
    const counts = { Low: 0, Medium: 0, High: 0 };
    STUDENTS.forEach((s) => (counts[s.riskBand] += 1));
    const needsAttention = [...STUDENTS].sort((a, b) => b.riskScore - a.riskScore).slice(0, 5);
    const byCourse = COURSES.map((c) => {
      const inCourse = STUDENTS.filter((s) => s.courseCode === c.code);
      return {
        course: c,
        total: inCourse.length,
        high: inCourse.filter((s) => s.riskBand === "High").length,
        medium: inCourse.filter((s) => s.riskBand === "Medium").length,
        low: inCourse.filter((s) => s.riskBand === "Low").length,
      };
    });
    return { counts, total: STUDENTS.length, needsAttention, byCourse };
  }
  const r = await fetch(`${API_BASE}/api/v1/overview`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  if (USE_MOCK) {
    await delay(200);
    assertNotForcedError();
    return MODEL_INFO;
  }
  const r = await fetch(`${API_BASE}/api/v1/model/info`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  const raw = await r.json();
  return {
    modelVersion: raw.model_version,
    trainedAt: raw.trained_at,
    checkpointFraction: raw.checkpoint_fraction,
    nTrainingRows: raw.n_training_rows,
    heldOutMetrics: raw.held_out_metrics,
  };
}
