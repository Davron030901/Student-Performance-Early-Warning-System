import { useQuery } from "@tanstack/react-query";
import type { StudentQuery } from "@/types";
import { fetchCourses, fetchModelInfo, fetchOverview, fetchStudentDetail, fetchStudents } from "./client";

export const useCourses = () =>
  useQuery({ queryKey: ["courses"], queryFn: fetchCourses, staleTime: Infinity });

export const useStudents = (query: StudentQuery) =>
  useQuery({ queryKey: ["students", query], queryFn: () => fetchStudents(query), placeholderData: (prev) => prev });

export const useStudentDetail = (id: string | undefined) =>
  useQuery({ queryKey: ["student", id], queryFn: () => fetchStudentDetail(id!), enabled: Boolean(id) });

export const useOverview = () => useQuery({ queryKey: ["overview"], queryFn: fetchOverview });

export const useModelInfo = () =>
  useQuery({ queryKey: ["modelInfo"], queryFn: fetchModelInfo, staleTime: Infinity });
