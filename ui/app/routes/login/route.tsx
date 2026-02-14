import type { LoaderFunctionArgs } from "react-router";
import { redirect } from "react-router";
import {
  buildGithubAuthorizeUrl,
  createOAuthCookie,
  createPkceVerifier,
  generateState,
} from "~/utils/auth.server";

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const rawReturnTo = url.searchParams.get("returnTo") ?? "/";
  const returnTo = rawReturnTo.startsWith("/") ? rawReturnTo : "/";

  const state = generateState();
  const { verifier, challenge } = createPkceVerifier();

  const authorizeUrl = buildGithubAuthorizeUrl({
    state,
    codeChallenge: challenge,
  });

  return redirect(authorizeUrl, {
    headers: {
      "Set-Cookie": createOAuthCookie(state, verifier, returnTo),
    },
  });
}
