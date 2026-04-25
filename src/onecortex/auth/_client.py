from .._http import HttpClient
from .models import AuthSession, AuthUser


class AuthClient:
    """Client for the Onecortex auth service."""

    def __init__(self, http: HttpClient, base_path: str = "/auth"):
        self._http = http
        self._base_path = base_path

    def sign_up(self, email: str, password: str, metadata: dict | None = None) -> AuthSession:
        """Register a new user with email and password."""
        body: dict = {"email": email, "password": password}
        if metadata:
            body["data"] = metadata
        response = self._http.post(f"{self._base_path}/signup", json=body)
        session = AuthSession.model_validate(response.json())
        self._http.set_token(session.access_token, session.refresh_token, session.expires_at)
        return session

    def login(self, email: str, password: str) -> AuthSession:
        """Authenticate with email and password, returning a session."""
        body = {"grant_type": "password", "email": email, "password": password}
        response = self._http.post(f"{self._base_path}/token", json=body)
        session = AuthSession.model_validate(response.json())
        self._http.set_token(session.access_token, session.refresh_token, session.expires_at)
        return session

    def logout(self) -> None:
        """Invalidate the current session token."""
        self._http.post(f"{self._base_path}/logout")
        self._http.clear_token()

    def refresh(self) -> AuthSession:
        """Exchange the current refresh token for a new session."""
        if not self._http._refresh_token:
            from ..exceptions import UnauthorizedError
            raise UnauthorizedError("No refresh token available. Call login() first.")
        body = {"grant_type": "refresh_token", "refresh_token": self._http._refresh_token}
        response = self._http.post(f"{self._base_path}/token", json=body)
        session = AuthSession.model_validate(response.json())
        self._http.set_token(session.access_token, session.refresh_token, session.expires_at)
        return session

    def get_user(self) -> AuthUser:
        """Fetch the current user's profile."""
        response = self._http.get(f"{self._base_path}/user")
        return AuthUser.model_validate(response.json())

    def update_user(
        self,
        email: str | None = None,
        phone: str | None = None,
        password: str | None = None,
        data: dict | None = None,
    ) -> AuthUser:
        """Update the current user's profile."""
        body: dict = {}
        if email is not None:
            body["email"] = email
        if phone is not None:
            body["phone"] = phone
        if password is not None:
            body["password"] = password
        if data is not None:
            body["data"] = data
        response = self._http.put(f"{self._base_path}/user", json=body)
        return AuthUser.model_validate(response.json())

    def request_magic_link(self, email: str) -> None:
        """Send a magic link to the given email address."""
        self._http.post(f"{self._base_path}/magiclink", json={"email": email})

    def verify_otp(
        self,
        token: str,
        type: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> AuthSession:
        """Verify a one-time token (OTP or magic link token)."""
        body: dict = {"token": token, "type": type}
        if email is not None:
            body["email"] = email
        if phone is not None:
            body["phone"] = phone
        response = self._http.post(f"{self._base_path}/verify", json=body)
        session = AuthSession.model_validate(response.json())
        self._http.set_token(session.access_token, session.refresh_token, session.expires_at)
        return session

    def set_session(
        self,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        """Manually set a pre-obtained session token (server-side use)."""
        self._http.set_token(access_token, refresh_token, expires_at)
