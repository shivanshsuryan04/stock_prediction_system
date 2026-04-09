"use client";

import { useEffect, useRef } from "react";
import { useAuthStore } from "@/lib/auth.store";

export function AuthHydrator() {
  const hydrate     = useAuthStore((s) => s.hydrate);
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      hydrate();
    }
  }, [hydrate]);

  return null;
}