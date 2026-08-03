import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { DashboardPage } from "@/features/dashboard/DashboardPage";
import { RosterPage } from "@/features/roster/RosterPage";
import { StudentDetailPage } from "@/features/student-detail/StudentDetailPage";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Render's free instance sleeps after ~15 minutes idle and takes roughly
      // 50 seconds to wake. Without generous retries, the first visit after a
      // quiet period shows an error page even though nothing is broken. Three
      // attempts with backoff covers a cold start; genuine failures still
      // surface, just a little later.
      retry: 3,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 15000),
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "students", element: <RosterPage /> },
      { path: "students/:id", element: <StudentDetailPage /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>
);
