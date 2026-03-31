import { Router } from "express";
import { body } from "express-validator";
import { register, login, refreshToken, logout, getMe } from "../controllers/auth.controller";
import { authenticate } from "../middleware/auth.middleware";

const router = Router();

const registerValidation = [
  body("name").trim().notEmpty().withMessage("Name is required.").isLength({ min: 2, max: 50 }),
  body("email").isEmail().normalizeEmail().withMessage("A valid email is required."),
  body("password")
    .isLength({ min: 8 }).withMessage("Password must be at least 8 characters.")
    .matches(/^(?=.*[A-Z])(?=.*[0-9])/).withMessage("Password must contain at least one uppercase letter and one number."),
];

const loginValidation = [
  body("email").isEmail().normalizeEmail().withMessage("A valid email is required."),
  body("password").notEmpty().withMessage("Password is required."),
];

router.post("/register", registerValidation, register);
router.post("/login", loginValidation, login);
router.post("/refresh", refreshToken);
router.post("/logout", logout);
router.get("/me", authenticate, getMe);

export default router;