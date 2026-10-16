"""Password hashing, JWT tokens and role-based access control."""
import datetime as dt
import hashlib
import hmac
import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import User, get_db

ROLES = ("admin", "researcher", "clinician")
_ITERATIONS = 200_000
bearer = HTTPBearer(auto_error=False)


def hash_password(password):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "pbkdf2$%d$%s$%s" % (_ITERATIONS, salt.hex(), digest.hex())


def verify_password(password, stored):
    try:
        _, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(check.hex(), digest)


def create_token(user):
    payload = {
        "sub": str(user.id), "email": user.email, "role": user.role,
        "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=settings.jwt_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)):
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        payload = jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


def require_roles(*roles):
    def checker(user: User = Depends(current_user)):
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Requires role: %s" % " or ".join(roles))
        return user
    return checker


def ensure_admin(db):
    """Create the bootstrap admin account from ADMIN_EMAIL / ADMIN_PASSWORD if absent."""
    if not settings.admin_password:
        return
    if db.query(User).filter(User.email == settings.admin_email).first() is None:
        db.add(User(email=settings.admin_email, password_hash=hash_password(settings.admin_password), role="admin"))
        db.commit()
