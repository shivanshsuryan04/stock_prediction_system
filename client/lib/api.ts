import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import type { AuthResponse, RefreshResponse, User } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000/api";

export const api = axios.create({
  baseURL: API_URL,
  withCredentials: true,
  timeout: 15_000,
});

let accessToken: string | null = null;
export const setAccessToken = (token: string | null): void => { accessToken = token; };
export const getAccessToken = (): string | null => accessToken;

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  if (accessToken) config.headers.Authorization = `Bearer ${accessToken}`;
  return config;
});

type FailedRequest = { resolve: (token: string) => void; reject: (err: unknown) => void };
let isRefreshing = false;
let failedQueue: FailedRequest[] = [];

const processQueue = (error: unknown, token: string | null = null): void => {
  failedQueue.forEach((p) => (error ? p.reject(error) : p.resolve(token!)));
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    const isRefreshRequest = original.url?.includes("/auth/refresh");

    // THE LOOP CUTTER: Stop immediately if the refresh call itself failed
    if (error.response?.status === 401 && isRefreshRequest) {
      console.error("Refresh failed. Cutting loop to prevent 429.");
      isRefreshing = false;
      processQueue(error, null);
      setAccessToken(null);
      
      // Force redirect to break the React cycle
      if (typeof window !== "undefined" && !window.location.pathname.startsWith('/auth')) {
        window.location.href = "/auth/login";
      }
      return Promise.reject(error);
    }

    // Standard 401 retry logic for normal requests (like /auth/me)
    if (error.response?.status === 401 && !original._retry) {
      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          original.headers.Authorization = `Bearer ${token}`;
          return api(original);
        });
      }

      original._retry = true;
      isRefreshing = true;

      try {
        // Use base axios here to skip this interceptor for the refresh call itself
        const { data } = await axios.post<RefreshResponse>(
          `${API_URL}/auth/refresh`,
          {},
          { withCredentials: true }
        );
        
        setAccessToken(data.accessToken);
        processQueue(null, data.accessToken);
        
        original.headers.Authorization = `Bearer ${data.accessToken}`;
        return api(original);
      } catch (refreshError) {
        processQueue(refreshError, null);
        setAccessToken(null);
        if (typeof window !== "undefined" && !window.location.pathname.startsWith('/auth')) {
          window.location.href = "/auth/login";
        }
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// ---------------------------------------------------------
// API Services
// ---------------------------------------------------------

export const authApi = {
  register: (data: { name: string; email: string; password: string }) =>
    api.post<AuthResponse>("/auth/register", data),
  login: (data: { email: string; password: string }) =>
    api.post<AuthResponse>("/auth/login", data),
  logout: () => api.post<{ success: boolean }>("/auth/logout"),
  getMe: () => api.get<{ success: boolean; user: User }>("/auth/me"),
  refresh: () => api.post<RefreshResponse>("/auth/refresh"),
};

// Added Watchlist API using your custom authenticated interceptor
export const watchlistApi = {
  get: () => api.get("/watchlist"),
  add: (ticker: string) => api.post("/watchlist", { ticker }),
  remove: (ticker: string) => api.delete(`/watchlist/${ticker}`),
};

// Added Predictions API using your custom authenticated interceptor
export const predictionsApi = {
  getAll: () => api.get("/predictions"),
};