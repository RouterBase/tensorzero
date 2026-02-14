import crypto from "crypto";
import { getEnv } from "./env.server";

const SESSION_COOKIE_NAME = "tz_session";
const OAUTH_COOKIE_NAME = "tz_oauth";

const SESSION_TTL_SECONDS = 60 * 60 * 12;
const OAUTH_TTL_SECONDS = 10 * 60;

export type AuthUser = {
  id: number;
  login: string;
  name: string | null;
  avatarUrl: string;
  orgs: string[];
};

type SessionPayload = {
  exp: number;
  user: AuthUser;
};

type OAuthPayload = {
  exp: number;
  state: string;
  verifier: string;
  returnTo: string;
};

export type AuthConfig = {
  clientId: string;
  clientSecret: string;
  callbackUrl: string;
  allowedOrgs: string[];
  allowedTeams: string[];
  sessionSecret: string;
};

export function isAuthEnabled(): boolean {
  const env = getEnv();
  return !!(
    env.GITHUB_CLIENT_ID &&
    env.GITHUB_CLIENT_SECRET &&
    env.GITHUB_CALLBACK_URL &&
    env.GITHUB_ALLOWED_ORGS &&
    env.SESSION_SECRET
  );
}

function getAuthConfig(): AuthConfig {
  const env = getEnv();
  const clientId = env.GITHUB_CLIENT_ID;
  const clientSecret = env.GITHUB_CLIENT_SECRET;
  const callbackUrl = env.GITHUB_CALLBACK_URL;
  const allowedOrgs = env.GITHUB_ALLOWED_ORGS;
  const allowedTeams = env.GITHUB_ALLOWED_TEAMS;
  const sessionSecret = env.SESSION_SECRET;

  if (!clientId || !clientSecret || !callbackUrl || !allowedOrgs) {
    throw new Error(
      "Missing GitHub OAuth App env. Required: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GITHUB_CALLBACK_URL, GITHUB_ALLOWED_ORGS.",
    );
  }
  if (!sessionSecret || sessionSecret.length < 32) {
    throw new Error("SESSION_SECRET must be set and at least 32 chars.");
  }

  return {
    clientId,
    clientSecret,
    callbackUrl,
    allowedOrgs: allowedOrgs
      .split(",")
      .map((org) => org.trim())
      .filter(Boolean),
    allowedTeams: allowedTeams
      ? allowedTeams
          .split(",")
          .map((team) => team.trim())
          .filter(Boolean)
      : [],
    sessionSecret,
  };
}

function base64UrlEncode(input: Buffer | string): string {
  const buffer = typeof input === "string" ? Buffer.from(input) : input;
  return buffer
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function base64UrlDecode(input: string): Buffer {
  const padded = input.replace(/-/g, "+").replace(/_/g, "/");
  const padding =
    padded.length % 4 === 0 ? "" : "=".repeat(4 - (padded.length % 4));
  return Buffer.from(`${padded}${padding}`, "base64");
}

function hmacSign(payload: string, secret: string): string {
  return base64UrlEncode(
    crypto.createHmac("sha256", secret).update(payload).digest(),
  );
}

function signPayload(payload: object, secret: string): string {
  const serialized = JSON.stringify(payload);
  const encoded = base64UrlEncode(serialized);
  const signature = hmacSign(encoded, secret);
  return `${encoded}.${signature}`;
}

function verifyPayload<T>(value: string, secret: string): T | null {
  const [encoded, signature] = value.split(".");
  if (!encoded || !signature) return null;
  const expected = hmacSign(encoded, secret);
  if (signature.length !== expected.length) {
    return null;
  }
  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) {
    return null;
  }
  try {
    return JSON.parse(base64UrlDecode(encoded).toString("utf-8")) as T;
  } catch {
    return null;
  }
}

function parseCookies(request: Request): Record<string, string> {
  const header = request.headers.get("cookie");
  if (!header) return {};
  return header.split(";").reduce<Record<string, string>>((acc, part) => {
    const [rawKey, ...rest] = part.trim().split("=");
    if (!rawKey) return acc;
    acc[rawKey] = decodeURIComponent(rest.join("="));
    return acc;
  }, {});
}

function buildCookie(
  name: string,
  value: string,
  options: {
    maxAge?: number;
  } = {},
) {
  const env = getEnv();
  const base = `${name}=${encodeURIComponent(value)}; Path=/; HttpOnly; SameSite=Lax`;
  const secure = env.NODE_ENV === "production" ? "; Secure" : "";
  const maxAge =
    options.maxAge !== undefined ? `; Max-Age=${options.maxAge}` : "";
  return `${base}${maxAge}${secure}`;
}

export function getSession(request: Request): SessionPayload | null {
  const { sessionSecret } = getAuthConfig();
  const cookies = parseCookies(request);
  const raw = cookies[SESSION_COOKIE_NAME];
  if (!raw) return null;
  const payload = verifyPayload<SessionPayload>(raw, sessionSecret);
  if (!payload) return null;
  if (payload.exp < Math.floor(Date.now() / 1000)) {
    return null;
  }
  return payload;
}

export function createSessionCookie(user: AuthUser): string {
  const { sessionSecret } = getAuthConfig();
  const payload: SessionPayload = {
    exp: Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS,
    user,
  };
  return buildCookie(SESSION_COOKIE_NAME, signPayload(payload, sessionSecret), {
    maxAge: SESSION_TTL_SECONDS,
  });
}

export function clearSessionCookie(): string {
  return buildCookie(SESSION_COOKIE_NAME, "", { maxAge: 0 });
}

export function createOAuthCookie(
  state: string,
  verifier: string,
  returnTo: string,
): string {
  const { sessionSecret } = getAuthConfig();
  const payload: OAuthPayload = {
    exp: Math.floor(Date.now() / 1000) + OAUTH_TTL_SECONDS,
    state,
    verifier,
    returnTo,
  };
  return buildCookie(OAUTH_COOKIE_NAME, signPayload(payload, sessionSecret), {
    maxAge: OAUTH_TTL_SECONDS,
  });
}

export function consumeOAuthCookie(
  request: Request,
  state: string,
): { verifier: string; returnTo: string } | null {
  const { sessionSecret } = getAuthConfig();
  const cookies = parseCookies(request);
  const raw = cookies[OAUTH_COOKIE_NAME];
  if (!raw) return null;
  const payload = verifyPayload<OAuthPayload>(raw, sessionSecret);
  if (!payload) return null;
  if (payload.exp < Math.floor(Date.now() / 1000)) {
    return null;
  }
  if (payload.state !== state) {
    return null;
  }
  return { verifier: payload.verifier, returnTo: payload.returnTo };
}

export function clearOAuthCookie(): string {
  return buildCookie(OAUTH_COOKIE_NAME, "", { maxAge: 0 });
}

export function buildGithubAuthorizeUrl({
  state,
  codeChallenge,
}: {
  state: string;
  codeChallenge: string;
}): string {
  const { clientId, callbackUrl } = getAuthConfig();
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", callbackUrl);
  url.searchParams.set("state", state);
  url.searchParams.set("scope", "read:org");
  url.searchParams.set("code_challenge", codeChallenge);
  url.searchParams.set("code_challenge_method", "S256");
  return url.toString();
}

export function createPkceVerifier(): {
  verifier: string;
  challenge: string;
} {
  const verifier = base64UrlEncode(crypto.randomBytes(32));
  const challenge = base64UrlEncode(
    crypto.createHash("sha256").update(verifier).digest(),
  );
  return { verifier, challenge };
}

export function generateState(): string {
  return base64UrlEncode(crypto.randomBytes(24));
}

export function getAllowedOrgs(): string[] {
  return getAuthConfig().allowedOrgs.map((org) => org.toLowerCase());
}

export function getAllowedTeams(): string[] {
  return getAuthConfig().allowedTeams.map((team) => team.toLowerCase());
}

export function getAuthConfigSummary() {
  const { clientId, callbackUrl, allowedOrgs, allowedTeams } = getAuthConfig();
  return { clientId, callbackUrl, allowedOrgs, allowedTeams };
}

export function getGithubOAuthConfig() {
  const { clientId, clientSecret, callbackUrl } = getAuthConfig();
  return { clientId, clientSecret, callbackUrl };
}
