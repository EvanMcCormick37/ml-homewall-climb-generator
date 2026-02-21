import axios from "axios";

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
});

// Optional: Add interceptors for error handling, auth, etc.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Centralized error handling
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  },
);

export function is502(err: unknown): boolean {
  if (err && typeof err === "object") {
    // Axios error shape
    const axiosErr = err as {
      response?: { status?: number };
      message?: string;
    };
    if (axiosErr.response?.status === 502) return true;
    // Fallback: check message string
    if (axiosErr.message && /502/.test(axiosErr.message)) return true;
  }
  return false;
}
