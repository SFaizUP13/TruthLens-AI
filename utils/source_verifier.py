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
