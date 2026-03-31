import { Request, Response, NextFunction } from "express";
import { getPrediction, getAllPredictions, SUPPORTED_TICKERS } from "../services/ml.service";

export const getAll = async (
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const predictions = await getAllPredictions();
    res.status(200).json({ success: true, data: predictions });
  } catch (err) {
    next(err);
  }
};

export const getOne = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const ticker = req.params.ticker?.toUpperCase();
    const prediction = await getPrediction(ticker);
    res.status(200).json({ success: true, data: prediction });
  } catch (err) {
    next(err);
  }
};

export const getTickers = (_req: Request, res: Response): void => {
  res.status(200).json({ success: true, data: SUPPORTED_TICKERS });
};