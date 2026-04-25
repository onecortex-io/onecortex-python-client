from pydantic import BaseModel


class AuthUser(BaseModel):
    id: str
    email: str | None = None
    phone: str | None = None
    email_confirmed_at: str | None = None
    app_metadata: dict = {}
    user_metadata: dict = {}
    is_anonymous: bool = False
    created_at: str = ""
    updated_at: str = ""


class AuthSession(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 0
    expires_at: int | None = None
    refresh_token: str | None = None
    user: AuthUser
