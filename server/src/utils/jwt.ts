import jwt from "jsonwebtoken";
import { JwtPayload, CookieOptions } from "../types";

const ACCESS_TOKEN_SECRET = process.env.JWT_ACCESS_SECRET;
const REFRESH_TOKEN_SECRET = process.env.JWT_REFRESH_SECRET;

if (!ACCESS_TOKEN_SECRET || !REFRESH_TOKEN_SECRET) {
  throw new Error("JWT secrets are not set in environment variables!");
}

export const signAccessToken = (payload: JwtPayload): string =>
  jwt.sign(payload, ACCESS_TOKEN_SECRET, { expiresIn: "15m" });

export const signRefreshToken = (payload: JwtPayload): string =>
  jwt.sign(payload, REFRESH_TOKEN_SECRET, { expiresIn: "7d" });

export const verifyAccessToken = (token: string): JwtPayload =>
  jwt.verify(token, ACCESS_TOKEN_SECRET) as JwtPayload;

export const verifyRefreshToken = (token: string): JwtPayload =>
  jwt.verify(token, REFRESH_TOKEN_SECRET) as JwtPayload;

export const REFRESH_COOKIE_OPTIONS: CookieOptions = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "strict",
  maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days in ms
  path: "/api/auth",
};