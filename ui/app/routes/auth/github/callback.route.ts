import type { LoaderFunctionArgs } from "react-router";
import { redirect } from "react-router";
import {
  clearOAuthCookie,
  clearSessionCookie,
  consumeOAuthCookie,
  createSessionCookie,
  getAllowedOrgs,
  getAllowedTeams,
} from "~/utils/auth.server";
import { getGithubOAuthConfig } from "~/utils/auth.server";
import { logger } from "~/utils/logger";

const GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token";
const GITHUB_USER_URL = "https://api.github.com/user";
const GITHUB_ORGS_URL = "https://api.github.com/user/orgs?per_page=100";
const GITHUB_TEAMS_URL = "https://api.github.com/user/teams?per_page=100";

type GithubUser = {
  id: number;
  login: string;
  name: string | null;
  avatar_url: string;
};

type GithubOrg = {
  login: string;
};

type GithubTeam = {
  slug: string;
  organization: {
    login: string;
  };
};

async function exchangeToken(code: string, verifier: string) {
  const { clientId, clientSecret, callbackUrl } = getGithubOAuthConfig();

  const response = await fetch(GITHUB_TOKEN_URL, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      client_id: clientId,
      client_secret: clientSecret,
      code,
      redirect_uri: callbackUrl,
      code_verifier: verifier,
    }),
  });

  const body = (await response.json()) as {
    access_token?: string;
    error?: string;
    error_description?: string;
  };

  if (!response.ok || !body.access_token) {
    const message = body.error_description || body.error || "OAuth failed";
    throw new Error(message);
  }

  return body.access_token;
}

async function fetchGithubUser(accessToken: string): Promise<GithubUser> {
  const response = await fetch(GITHUB_USER_URL, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "tensorzero-ui",
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch GitHub user");
  }
  return (await response.json()) as GithubUser;
}

async function fetchGithubOrgs(accessToken: string): Promise<GithubOrg[]> {
  const response = await fetch(GITHUB_ORGS_URL, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "tensorzero-ui",
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch GitHub orgs");
  }
  return (await response.json()) as GithubOrg[];
}

async function fetchGithubTeams(accessToken: string): Promise<GithubTeam[]> {
  const response = await fetch(GITHUB_TEAMS_URL, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "tensorzero-ui",
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch GitHub teams");
  }
  return (await response.json()) as GithubTeam[];
}

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");

  if (!code || !state) {
    return new Response("Missing OAuth code/state", { status: 400 });
  }

  const oauthData = consumeOAuthCookie(request, state);
  if (!oauthData) {
    return new Response("Invalid OAuth state", { status: 400 });
  }

  try {
    const accessToken = await exchangeToken(code, oauthData.verifier);
    const [user, orgs] = await Promise.all([
      fetchGithubUser(accessToken),
      fetchGithubOrgs(accessToken),
    ]);

    const allowed = getAllowedOrgs();
    const allowedTeams = getAllowedTeams();
    const orgNames = orgs.map((org) => org.login.toLowerCase());
    const isAllowed = allowed.some((org) => orgNames.includes(org));
    logger.warn(
      `Auth check: user=${user.login}, userOrgs=[${orgNames.join(", ")}], allowedOrgs=[${allowed.join(", ")}], isAllowed=${isAllowed}`,
    );
    if (!isAllowed) {
      return new Response("User is not in an allowed organization", {
        status: 403,
        headers: {
          "Set-Cookie": clearOAuthCookie(),
        },
      });
    }

    if (allowedTeams.length > 0) {
      const teams = await fetchGithubTeams(accessToken);
      const teamNames = teams.map((team) =>
        `${team.organization.login}/${team.slug}`.toLowerCase(),
      );
      const isTeamAllowed = allowedTeams.some((team) =>
        teamNames.includes(team),
      );
      if (!isTeamAllowed) {
        return new Response("User is not in an allowed team", {
          status: 403,
          headers: {
            "Set-Cookie": clearOAuthCookie(),
          },
        });
      }
    }

    const userSession = {
      id: user.id,
      login: user.login,
      name: user.name,
      avatarUrl: user.avatar_url,
      orgs: orgs.map((org) => org.login),
    };

    const successHeaders = new Headers();
    successHeaders.append("Set-Cookie", clearOAuthCookie());
    successHeaders.append("Set-Cookie", createSessionCookie(userSession));

    return redirect(oauthData.returnTo || "/", {
      headers: successHeaders,
    });
  } catch {
    const failHeaders = new Headers();
    failHeaders.append("Set-Cookie", clearOAuthCookie());
    failHeaders.append("Set-Cookie", clearSessionCookie());

    return new Response("Authentication failed", {
      status: 401,
      headers: failHeaders,
    });
  }
}
