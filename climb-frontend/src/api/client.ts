import axios from "axios";

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
});

let getTokenFn: (() => Promise<string | null>) | null = null;

export function setAuthTokenProvider(fn: () => Promise<string | null>) {
  getTokenFn = fn;
}

apiClient.interceptors.request.use(async (config) => {
  let token: string | null = null;

  if (getTokenFn) {
    token = await getTokenFn();
  }

  // Fallback: use Clerk's global instance for requests (e.g. route loaders) that
  // fire before React has rendered AuthInit and registered getTokenFn.
  if (!token) {
    const w = window as { Clerk?: { session?: { getToken: () => Promise<string | null> } } };
    token = (await w.Clerk?.session?.getToken()) ?? null;
  }

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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
    const status = axiosErr.response?.status;
    if (status === 502 || status === 503) return true;
    // Fallback: check message string
    if (axiosErr.message && /50[23]/.test(axiosErr.message)) return true;
  }
  return false;
}
