import { Request, Response, NextFunction } from "express";
import { verifyAccessToken } from "../utils/jwt";
import { AppError } from "./errorHandler";
import prisma from "../config/db";
import { AuthenticatedRequest } from "../types";
import { Role } from "@prisma/client";

export const authenticate = async (
  req: Request,
  _res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader?.startsWith("Bearer ")) {
      throw new AppError("Authentication required. No token provided.", 401);
    }

    const token = authHeader.split(" ")[1];
    const decoded = verifyAccessToken(token);

    const user = await prisma.user.findUnique({
      where: { id: decoded.id },
      select: { id: true, email: true, name: true, role: true },
    });

    if (!user) {
      throw new AppError("The user belonging to this token no longer exists.", 401);
    }

    (req as AuthenticatedRequest).user = user;
    next();
  } catch (err) {
    next(err);
  }
};

export const authorize = (...roles: Role[]) => {
  return (req: Request, _res: Response, next: NextFunction): void => {
    const authedReq = req as AuthenticatedRequest;
    if (!roles.includes(authedReq.user?.role)) {
      return next(new AppError("You do not have permission to perform this action.", 403));
    }
    next();
  };
};