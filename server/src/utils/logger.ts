import { createLogger, format, transports, Logger } from "winston";

const { combine, timestamp, printf, colorize, errors } = format;

const devFormat = combine(
  colorize(),
  timestamp({ format: "HH:mm:ss" }),
  errors({ stack: true }),
  printf(({ level, message, timestamp: ts, stack }) => {
    return `${ts} [${level}]: ${(stack ?? message) as string}`;
  })
);

const prodFormat = combine(timestamp(), errors({ stack: true }), format.json());

const logger: Logger = createLogger({
  level: process.env.LOG_LEVEL ?? "info",
  format: process.env.NODE_ENV === "production" ? prodFormat : devFormat,
  transports: [
    new transports.Console(),
    ...(process.env.NODE_ENV === "production"
      ? [new transports.File({ filename: "logs/error.log", level: "error" })]
      : []),
  ],
});

export { logger };