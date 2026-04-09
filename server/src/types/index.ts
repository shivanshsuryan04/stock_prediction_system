/**
 * @file types/index.ts
 * @description Shared TypeScript interfaces and type augmentations.
 */

import { Request } from "express";
import { Role } from "@prisma/client";

// ============================================================
// AUGMENT EXPRESS REQUEST — add typed `user` to req
// ============================================================
export interface AuthenticatedUser {
  id: string;
  email: string;
  name: string;
  role: Role;
}

export interface AuthenticatedRequest extends Request {
  user: AuthenticatedUser;
}

// ============================================================
// JWT PAYLOAD
// ============================================================
export interface JwtPayload {
  id: string;
  email: string;
  role: Role;
  iat?: number;
  exp?: number;
}

// ============================================================
// ML / PREDICTION TYPES
// ============================================================
export interface PredictionResult {
  ticker: string;
  xgbSignal: string;
  lstmSignal: string;
  lstmConf: number;
  finalSignal: string;
  cachedAt: Date;
  fromCache: boolean;
}

export interface PredictionEntry {
  ticker: string;
  data: PredictionResult | null;
  error: string | null;
}

/** Shape returned by the Python FastAPI /predict/:ticker endpoint */
export interface PythonApiPrediction {
  ticker: string;
  xgbSignal: string;
  lstmSignal: string;
  lstmConf: number;
  finalSignal: string;
  // Add these if you want to use the extra metadata
  xgbProbUp?: number;
  xgbProbDown?: number;
  fromCache?: boolean;
  cachedAt?: string;
}

// ============================================================
// API RESPONSE WRAPPERS
// ============================================================
export interface ApiResponse<T = unknown> {
  success: boolean;
  message?: string;
  data?: T;
}

export interface ApiErrorResponse {
  success: false;
  message: string;
  errors?: Array<{ msg: string; path: string }>;
}

// ============================================================
// COOKIE OPTIONS TYPE
// ============================================================
export interface CookieOptions {
  httpOnly: boolean;
  secure: boolean;
  sameSite: "strict" | "lax" | "none";
  maxAge: number;
  path: string;
}