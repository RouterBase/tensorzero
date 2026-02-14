import { redirect } from "react-router";
import type { Route } from "../+types/root";
import { getSession, isAuthEnabled } from "./auth.server";

const AUTH_EXEMPT_PATHS = new Set([
  "/login",
  "/logout",
  "/auth/github/callback",
  "/health",
  "/favicon.svg",
]);

function isAssetPath(pathname: string): boolean {
  return (
    pathname.startsWith("/build/") ||
    pathname.startsWith("/assets/") ||
    pathname.startsWith("/@")
  );
}

export const authMiddleware: Route.MiddlewareFunction = async ({ request }) => {
  if (!isAuthEnabled()) return;

  const url = new URL(request.url);
  if (AUTH_EXEMPT_PATHS.has(url.pathname) || isAssetPath(url.pathname)) {
    return;
  }

  const session = getSession(request);
  if (session) return;

  if (url.pathname.startsWith("/api/")) {
    throw new Response("Unauthorized", { status: 401 });
  }

  const returnTo = `${url.pathname}${url.search}`;
  throw redirect(`/login?returnTo=${encodeURIComponent(returnTo)}`);
};
