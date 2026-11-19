//! Guard for fetching caller-supplied URLs (RouterBase).
//!
//! A file input given as a URL (`{"type": "image", "url": ...}`, OpenAI
//! `image_url`, ...) is downloaded by the gateway itself. Without a guard, any
//! API caller can make the gateway request `http://127.0.0.1:<port>/`, the
//! cloud metadata endpoint, or cluster-internal services, and read the result
//! back from the error message (server-side request forgery).
//!
//! Every such fetch goes through [`client`], which:
//! * only speaks `http`/`https`;
//! * resolves hostnames itself and drops any non-public address (so DNS
//!   rebinding and hostnames pointing at internal IPs are refused at connect
//!   time, not just at validation time);
//! * re-checks every redirect hop, since a literal-IP `Location` bypasses DNS;
//! * never uses a proxy.
//!
//! Set `TENSORZERO_ALLOW_PRIVATE_FILE_URLS=1` to turn the guard off (local
//! development against services on a private network only).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::{Client, redirect};
use url::{Host, Url};

use crate::endpoints::status::TENSORZERO_VERSION;

const MAX_REDIRECTS: usize = 5;
const FETCH_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Whether the guard is disabled by `TENSORZERO_ALLOW_PRIVATE_FILE_URLS`.
pub fn private_urls_allowed() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        std::env::var("TENSORZERO_ALLOW_PRIVATE_FILE_URLS")
            .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false)
    })
}

/// `true` only for globally routable unicast addresses.
pub fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_ipv4(v4),
        IpAddr::V6(v6) => is_public_ipv6(v6),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_multicast()
        || a == 0 // 0.0.0.0/8 "this network"
        || (a == 100 && (64..=127).contains(&b)) // 100.64.0.0/10 CGNAT (incl. Aliyun metadata 100.100.100.200)
        || (a == 192 && b == 0 && c == 0) // 192.0.0.0/24 IETF protocol assignments
        || (a == 198 && (b == 18 || b == 19)) // 198.18.0.0/15 benchmarking
        || a >= 240) // 240.0.0.0/4 reserved
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_public_ipv4(v4);
    }
    let seg = ip.segments();
    // 64:ff9b::/96 NAT64 — judge by the embedded IPv4 address.
    if seg[0] == 0x64 && seg[1] == 0xff9b && seg[2..6].iter().all(|s| *s == 0) {
        let [a, b] = seg[6].to_be_bytes();
        let [c, d] = seg[7].to_be_bytes();
        return is_public_ipv4(Ipv4Addr::new(a, b, c, d));
    }
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || seg[0..6].iter().all(|s| *s == 0) // ::/96 IPv4-compatible (deprecated)
        || (seg[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (seg[0] & 0xffc0) == 0xfe80 // fe80::/10 link local
        || (seg[0] & 0xffc0) == 0xfec0 // fec0::/10 site local (deprecated)
        || (seg[0] == 0x2001 && seg[1] == 0x0db8) // 2001:db8::/32 documentation
        || seg[0] == 0x0100 && seg[1..4].iter().all(|s| *s == 0)) // 100::/64 discard
}

/// Static checks on a URL before it is fetched: scheme, and the host when it
/// is an IP literal or a well-known loopback name. Hostnames are checked
/// again, by address, at connect time (see [`PublicOnlyResolver`]).
pub fn check_url(url: &Url) -> Result<(), String> {
    if private_urls_allowed() {
        return Ok(());
    }
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(format!("URL scheme `{other}` is not allowed")),
    }
    match url.host() {
        None => Err("URL has no host".to_string()),
        Some(Host::Ipv4(ip)) if !is_public_ipv4(ip) => {
            Err("URL points to a non-public address".to_string())
        }
        Some(Host::Ipv6(ip)) if !is_public_ipv6(ip) => {
            Err("URL points to a non-public address".to_string())
        }
        Some(Host::Domain(domain)) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            if domain == "localhost" || domain.ends_with(".localhost") {
                Err("URL points to a non-public address".to_string())
            } else {
                Ok(())
            }
        }
        Some(_) => Ok(()),
    }
}

/// DNS resolver that only hands back public addresses. A hostname that
/// resolves exclusively to internal addresses fails to connect.
#[derive(Debug)]
struct PublicOnlyResolver;

impl Resolve for PublicOnlyResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let host = name.as_str().to_string();
            let resolved = tokio::net::lookup_host((host.as_str(), 0)).await?;
            let public: Vec<SocketAddr> = resolved
                .filter(|addr| private_urls_allowed() || is_public_ip(addr.ip()))
                .collect();
            if public.is_empty() {
                return Err(format!("`{host}` does not resolve to a public address").into());
            }
            let addrs: Addrs = Box::new(public.into_iter());
            Ok(addrs)
        })
    }
}

/// The HTTP client for caller-supplied URLs. Built once.
pub fn client() -> Result<&'static Client, String> {
    static CLIENT: OnceLock<Result<Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            Client::builder()
                .timeout(FETCH_TIMEOUT)
                .connect_timeout(CONNECT_TIMEOUT)
                .user_agent(format!("TensorZero/{TENSORZERO_VERSION}"))
                .no_proxy()
                .dns_resolver(Arc::new(PublicOnlyResolver))
                .redirect(redirect::Policy::custom(|attempt| {
                    if attempt.previous().len() >= MAX_REDIRECTS {
                        return attempt.error("too many redirects");
                    }
                    match check_url(attempt.url()) {
                        Ok(()) => attempt.follow(),
                        Err(reason) => attempt.error(reason),
                    }
                }))
                .build()
                .map_err(|e| format!("Failed to build file-fetch HTTP client: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn internal_addresses_are_not_public() {
        for addr in [
            "127.0.0.1",
            "0.0.0.0",
            "10.109.0.1",
            "172.18.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.100.100.200",
            "100.64.0.1",
            "198.18.0.1",
            "255.255.255.255",
            "224.0.0.1",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a9fe:a9fe",
            "::127.0.0.1",
        ] {
            assert!(!is_public_ip(ip(addr)), "{addr} must be refused");
        }
    }

    #[test]
    fn public_addresses_are_allowed() {
        for addr in [
            "8.8.8.8",
            "1.1.1.1",
            "104.22.17.196",
            "2606:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public_ip(ip(addr)), "{addr} must be allowed");
        }
    }

    #[tokio::test]
    async fn resolver_drops_names_that_only_resolve_internally() {
        // `localhost` resolves to 127.0.0.1 / ::1 on every host. The resolver is
        // what stops a public-looking hostname (or a rebinding one) that points
        // at an internal address, since `check_url` only sees the name.
        let name: Name = "localhost".parse().unwrap();
        let result = PublicOnlyResolver.resolve(name).await;
        assert!(
            result.is_err(),
            "a name resolving only to loopback must not connect"
        );
    }

    #[tokio::test]
    async fn redirects_to_internal_addresses_are_not_followed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // Stand-in for a public host that answers with a redirect into the
        // cluster. (The first hop is called on the client directly, skipping
        // `check_url`, so the redirect policy is what's under test.)
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Test-only server; nothing waits on it at shutdown.
        #[expect(clippy::disallowed_methods)]
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let _ = sock
                    .write_all(
                        b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
            }
        });
        let err = client()
            .unwrap()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect_err("a redirect to the metadata address must not be followed");
        assert!(
            err.is_redirect(),
            "expected a redirect-policy error, got {err:?}"
        );
    }

    #[test]
    fn check_url_refuses_internal_literals_and_odd_schemes() {
        for url in [
            "http://127.0.0.1:3000/metrics",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://localhost:6379/",
            "http://foo.localhost/",
            "http://2130706433/",
            "http://0x7f000001/",
            "file:///etc/passwd",
            "gopher://example.com/",
        ] {
            let parsed = Url::parse(url).unwrap();
            assert!(check_url(&parsed).is_err(), "{url} must be refused");
        }
        for url in ["https://example.com/cat.png", "http://8.8.8.8/x.png"] {
            let parsed = Url::parse(url).unwrap();
            assert!(check_url(&parsed).is_ok(), "{url} must be allowed");
        }
    }
}
