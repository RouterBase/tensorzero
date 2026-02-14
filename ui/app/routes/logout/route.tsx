import { redirect } from "react-router";
import { clearSessionCookie } from "~/utils/auth.server";

export async function action() {
  return redirect("/login", {
    headers: {
      "Set-Cookie": clearSessionCookie(),
    },
  });
}

// Redirect GET requests to the home page
export async function loader() {
  return redirect("/");
}
