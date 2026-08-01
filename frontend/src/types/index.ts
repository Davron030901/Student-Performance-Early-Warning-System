/** Risk band as returned by the ML API. */
export type RiskBand = "Low" | "Medium" | "High";

/**
 * Advisor-facing labels. The API speaks in Low/Medium/High; the interface
 * speaks in terms of what the advisor should *do*. "Needs a check-in" is a
 * prompt to look closer — "High risk" sounds like a verdict about a student,
 * which is neither accurate (it is a probability) nor the tone this tool wants.
 */
export const RISK_LABEL: Record<RiskBand, string> = {
  Low: "Steady",
  Medium: "Worth a look",
  High: "Needs a check-in",
};

export interface Course {
  code: string;
  title: string;
  presentation: string;
  lengthDays: number;
  checkpointDay: number;
}

export interface WeeklyActivity {
  week: number;
  clicks: number;
  /** Whether this week falls at or before the prediction checkpoint. */
  beforeCheckpoint: boolean;
}

export interface StudentSummary {
  id: string;
  name: string;
  courseCode: string;
  riskScore: number;
  riskBand: RiskBand;
  submittedCount: number;
  expectedCount: number;
  lastActiveDaysAgo: number;
  activity: WeeklyActivity[];
}

export interface RiskFactor {
  text: string;
  /** Positive values push risk up, negative values pull it down. */
  impact: number;
}

export interface StudentDetail extends StudentSummary {
  topFactors: RiskFactor[];
  avgEarlyScore: number | null;
  onTimeRate: number;
  totalClicks: number;
  activeDays: number;
  registeredDay: number;
  checkpointUsed: string;
  modelVersion: string;
}

export interface StudentListResponse {
  students: StudentSummary[];
  total: number;
  page: number;
  pageSize: number;
}

export interface ModelInfo {
  modelVersion: string;
  trainedAt: string;
  checkpointFraction: number;
  nTrainingRows: number;
  heldOutMetrics: { recall: number; precision: number; f2: number; roc_auc: number };
}

export interface StudentQuery {
  course?: string;
  riskBand?: RiskBand | "all";
  search?: string;
  page?: number;
  sortBy?: "risk" | "name" | "lastActive";
}
