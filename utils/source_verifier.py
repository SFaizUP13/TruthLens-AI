from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse


def check_source_url(url, timeout=4):
    """
    Check basic URL validity and accessibility.

    This does not verify the truthfulness of a source.
    """

    if not isinstance(url, str) or not url.strip():
        return "INVALID_URL"

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "INVALID_URL"

    request = Request(
        url,
        headers={
            "User-Agent": "TruthLensAI/1.0"
        },
        method="HEAD"
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status

            if 200 <= status < 400:
                return "ACCESSIBLE"

            return f"HTTP_{status}"

    except HTTPError as error:
        if error.code in (404, 410):
            return f"HTTP_{error.code}"

        if error.code in (401, 403, 405, 429):
            return "ACCESSIBILITY_UNCONFIRMED"

        return f"HTTP_{error.code}"

    except (URLError, TimeoutError, OSError):
        return "ACCESSIBILITY_UNCONFIRMED"

from html.parser import HTMLParser
from urllib.parse import urljoin
import ipaddress
import socket


class _TextExtractor(HTMLParser):
    """Extract visible text from basic HTML."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        if self.skip_depth == 0:
            text = data.strip()
            if text:
                self.parts.append(text)

    def get_text(self):
        return " ".join(self.parts)


def _is_public_url(url):
    """Reject local/private network destinations."""

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if not parsed.hostname:
        return False

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            None,
            type=socket.SOCK_STREAM,
        )

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if not ip.is_global:
                return False

        return bool(addresses)

    except (OSError, ValueError):
        return False


def fetch_source_text(url, timeout=5, max_chars=12000):
    """
    Attempt to retrieve readable webpage text.

    Returns a dictionary containing:
    - status
    - text

    This does NOT establish whether the webpage
    supports a particular factual claim.
    """

    if not isinstance(url, str) or not url.strip():
        return {
            "status": "INVALID_URL",
            "text": "",
        }

    url = url.strip()

    if not _is_public_url(url):
        return {
            "status": "INVALID_OR_PRIVATE_URL",
            "text": "",
        }

    request = Request(
        url,
        headers={
            "User-Agent": "TruthLensAI/1.0",
            "Accept": "text/html",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:

            final_url = response.geturl()

            if not _is_public_url(final_url):
                return {
                    "status": "INVALID_OR_PRIVATE_URL",
                    "text": "",
                }

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:
                return {
                    "status": "UNSUPPORTED_CONTENT_TYPE",
                    "text": "",
                }

            # Limit download size to approximately 200 KB.
            raw_html = response.read(200000)

            charset = response.headers.get_content_charset()
            charset = charset or "utf-8"

            html = raw_html.decode(
                charset,
                errors="replace",
            )

            parser = _TextExtractor()
            parser.feed(html)

            text = parser.get_text()

            text = " ".join(text.split())

            if not text:
                return {
                    "status": "CONTENT_UNAVAILABLE",
                    "text": "",
                }

            return {
                "status": "CONTENT_RETRIEVED",
                "text": text[:max_chars],
            }

    except (HTTPError, URLError, TimeoutError, OSError):
        return {
            "status": "CONTENT_UNAVAILABLE",
            "text": "",
        }

    except Exception:
        return {
            "status": "CONTENT_UNAVAILABLE",
            "text": "",
        }
