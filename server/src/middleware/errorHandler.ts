import { Request, Response, NextFunction } from "express";
import { logger } from "../utils/logger";

export class AppError extends Error {
  public readonly statusCode: number;
  public readonly isOperational: boolean;

  constructor(message: string, statusCode: number) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}

interface PrismaError extends Error {
  code?: string;
  meta?: { target?: string[] };
}

export const errorHandler = (
  err: PrismaError & { statusCode?: number; isOperational?: boolean },
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  logger.error({
    message: err.message,
    stack: err.stack,
    url: req.originalUrl,
    method: req.method,
    ip: req.ip,
  });

  // Prisma: unique constraint violation
  if (err.code === "P2002") {
    res.status(409).json({
      success: false,
      message: `A record with this ${err.meta?.target?.[0] ?? "field"} already exists.`,
    });
    return;
  }

  // Prisma: record not found
  if (err.code === "P2025") {
    res.status(404).json({ success: false, message: "Record not found." });
    return;
  }

  // JWT errors
  if (err.name === "JsonWebTokenError") {
    res.status(401).json({ success: false, message: "Invalid token." });
    return;
  }
  if (err.name === "TokenExpiredError") {
    res.status(401).json({ success: false, message: "Token expired." });
    return;
  }

  // Known operational errors
  if (err.isOperational) {
    res.status(err.statusCode ?? 400).json({ success: false, message: err.message });
    return;
  }

  // Unknown errors — never expose details in production
  const message =
    process.env.NODE_ENV === "production"
      ? "An unexpected internal error occurred."
      : err.message;

  res.status(err.statusCode ?? 500).json({ success: false, message });
};