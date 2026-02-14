"use client";

import { createContext, use } from "react";

const AuthEnabledContext = createContext(false);
AuthEnabledContext.displayName = "AuthEnabledContext";

export function useAuthEnabled() {
  return use(AuthEnabledContext);
}

export function AuthEnabledProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: boolean;
}) {
  return <AuthEnabledContext value={value}>{children}</AuthEnabledContext>;
}
