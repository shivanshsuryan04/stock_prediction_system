import { Router } from "express";
import { getAll, getOne, getTickers } from "../controllers/prediction.controller";
import { authenticate } from "../middleware/auth.middleware";

const router = Router();

router.use(authenticate);

router.get("/", getAll);
router.get("/tickers", getTickers);
router.get("/:ticker", getOne);

export default router;