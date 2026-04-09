import { create } from "zustand";
import { authApi, setAccessToken } from "@/lib/api";
import type { User } from "@/types";

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  hydrate: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isLoading: true,
  isAuthenticated: false,

  login: async (email, password) => {
    const { data } = await authApi.login({ email, password });
    setAccessToken(data.accessToken);
    set({ user: data.user, isAuthenticated: true });
  },

  register: async (name, email, password) => {
    const { data } = await authApi.register({ name, email, password });
    setAccessToken(data.accessToken);
    set({ user: data.user, isAuthenticated: true });
  },

  logout: async () => {
    await authApi.logout().catch(() => {});
    setAccessToken(null);
    set({ user: null, isAuthenticated: false });
  },

  hydrate: async () => {
    try {
      set({ isLoading: true });
      // The interceptor handles the silent refresh automatically when getMe returns 401.
      const { data: meData } = await authApi.getMe();
      set({ user: meData.user, isAuthenticated: true });
    } catch (error) {
      setAccessToken(null);
      set({ user: null, isAuthenticated: false });
    } finally {
      set({ isLoading: false });
    }
  },
}));