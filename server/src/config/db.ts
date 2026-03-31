import { PrismaClient, Prisma } from "@prisma/client";
import { logger } from "../utils/logger";

declare global {
  // eslint-disable-next-line no-var
  var __prisma: PrismaClient<Prisma.PrismaClientOptions, "query" | "error" | "warn"> | undefined;
}

const prisma =
  globalThis.__prisma ??
  new PrismaClient<Prisma.PrismaClientOptions, "query" | "error" | "warn">({
    log: [
      { emit: "event", level: "query" },
      { emit: "event", level: "error" },
      { emit: "event", level: "warn" },
    ],
  });

prisma.$on("query", (e: Prisma.QueryEvent) => {
  if (e.duration > 500) {
    logger.warn(`Slow query (${e.duration}ms): ${e.query}`);
  }
});

prisma.$on("error", (e: Prisma.LogEvent) => {
  logger.error("Prisma error:", e);
});

if (process.env.NODE_ENV !== "production") {
  globalThis.__prisma = prisma;
}

export default prisma;