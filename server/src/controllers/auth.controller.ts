import { Request, Response, NextFunction } from "express";
import bcrypt from "bcryptjs";
import { validationResult } from "express-validator";
import prisma from "../config/db";
import {
  signAccessToken,
  signRefreshToken,
  verifyRefreshToken,
  REFRESH_COOKIE_OPTIONS,
} from "../utils/jwt";
import { AppError } from "../middleware/errorHandler";
import { AuthenticatedRequest, JwtPayload } from "../types";

export const register = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      res.status(422).json({ success: false, errors: errors.array() });
      return;
    }

    const { name, email, password } = req.body as {
      name: string;
      email: string;
      password: string;
    };

    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) throw new AppError("An account with this email already exists.", 409);

    const passwordHash = await bcrypt.hash(password, 12);

    const user = await prisma.user.create({
      data: { name, email, passwordHash },
      select: { id: true, name: true, email: true, role: true, createdAt: true },
    });

    const tokenPayload: JwtPayload = { id: user.id, email: user.email, role: user.role };
    const accessToken = signAccessToken(tokenPayload);
    const refreshToken = signRefreshToken(tokenPayload);

    res.cookie("refreshToken", refreshToken, REFRESH_COOKIE_OPTIONS);
    res.status(201).json({ success: true, message: "Account created.", accessToken, user });
  } catch (err) {
    next(err);
  }
};

export const login = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      res.status(422).json({ success: false, errors: errors.array() });
      return;
    }

    const { email, password } = req.body as { email: string; password: string };

    const user = await prisma.user.findUnique({ where: { email } });

    // Constant-time comparison to prevent timing attacks
    const dummyHash = "$2a$12$dummyhashtopreventtimingattacks123456";
    const passwordMatch = await bcrypt.compare(
      password,
      user ? user.passwordHash : dummyHash
    );

    if (!user || !passwordMatch) throw new AppError("Invalid email or password.", 401);

    const tokenPayload: JwtPayload = { id: user.id, email: user.email, role: user.role };
    const accessToken = signAccessToken(tokenPayload);
    const refreshToken = signRefreshToken(tokenPayload);

    res.cookie("refreshToken", refreshToken, REFRESH_COOKIE_OPTIONS);
    res.status(200).json({
      success: true,
      message: "Login successful.",
      accessToken,
      user: { id: user.id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    next(err);
  }
};

export const refreshToken = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const token = (req.cookies as Record<string, string>)?.refreshToken;
    if (!token) throw new AppError("No refresh token provided.", 401);

    const decoded = verifyRefreshToken(token);

    const user = await prisma.user.findUnique({
      where: { id: decoded.id },
      select: { id: true, email: true, role: true },
    });

    if (!user) throw new AppError("User not found.", 401);

    const newAccessToken = signAccessToken({ id: user.id, email: user.email, role: user.role });
    res.status(200).json({ success: true, accessToken: newAccessToken });
  } catch (err) {
    next(err);
  }
};

export const logout = (_req: Request, res: Response): void => {
  res.clearCookie("refreshToken", { 
    path: "/", 
    sameSite: "lax", 
    secure: process.env.NODE_ENV === "production", 
    httpOnly: true 
  });
  res.status(200).json({ success: true, message: "Logged out successfully." });
};

export const getMe = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const { id } = (req as AuthenticatedRequest).user;
    const user = await prisma.user.findUnique({
      where: { id },
      select: { id: true, name: true, email: true, role: true, createdAt: true },
    });
    res.status(200).json({ success: true, user });
  } catch (err) {
    next(err);
  }
};