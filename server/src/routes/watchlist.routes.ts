import { Router, Request, Response, NextFunction } from "express";
import { authenticate } from "../middleware/auth.middleware";
import prisma from "../config/db";
import { AppError } from "../middleware/errorHandler";
import { getPrediction } from "../services/ml.service";
import { AuthenticatedRequest } from "../types";

const router = Router();
router.use(authenticate);

// GET /api/watchlist
router.get("/", async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { id } = (req as AuthenticatedRequest).user;
    const items = await prisma.watchList.findMany({
      where: { userId: id },
      orderBy: { addedAt: "desc" },
    });

    const enriched = await Promise.allSettled(
      items.map(async (item) => {
        const prediction = await getPrediction(item.ticker).catch(() => null);
        return { ...item, prediction };
      })
    );

    const data = enriched
      .map((r) => (r.status === "fulfilled" ? r.value : null))
      .filter(Boolean);

    res.status(200).json({ success: true, data });
  } catch (err) {
    next(err);
  }
});

// POST /api/watchlist
router.post("/", async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { id } = (req as AuthenticatedRequest).user;
    const { ticker } = req.body as { ticker?: string };
    if (!ticker) throw new AppError("Ticker is required.", 400);

    const item = await prisma.watchList.create({
      data: { userId: id, ticker: ticker.toUpperCase() },
    });
    res.status(201).json({ success: true, data: item });
  } catch (err) {
    next(err);
  }
});

// DELETE /api/watchlist/:ticker
router.delete("/:ticker", async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { id } = (req as AuthenticatedRequest).user;
    await prisma.watchList.deleteMany({
      where: { userId: id, ticker: req.params.ticker?.toUpperCase() },
    });
    res.status(200).json({ success: true, message: "Removed from watchlist." });
  } catch (err) {
    next(err);
  }
});

export default router;