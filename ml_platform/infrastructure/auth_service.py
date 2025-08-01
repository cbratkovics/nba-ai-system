"""
Centralized Authentication Service with JWT, OAuth2, and Service-to-Service Auth
Includes permission management, session handling, and security features
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import redis
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge
import logging
from cryptography.fernet import Fernet
import pyotp
import qrcode
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import requests
import uuid

# Metrics
auth_attempts = Counter('auth_attempts_total', 'Authentication attempts', ['method', 'status'])
active_sessions = Gauge('active_sessions', 'Active user sessions')
token_operations = Counter('token_operations_total', 'Token operations', ['operation', 'type'])
permission_checks = Counter('permission_checks_total', 'Permission checks', ['resource', 'result'])

Base = declarative_base()
logger = logging.getLogger(__name__)


class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class AuthMethod(Enum):
    PASSWORD = "password"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    SERVICE_TOKEN = "service_token"
    MFA = "mfa"


@dataclass
class Permission:
    resource: str
    action: str
    scope: Optional[str] = None


# Database Models
class UserDB(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    username = Column(String, unique=True)
    password_hash = Column(String)
    role = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String)
    last_login = Column(DateTime)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class SessionDB(Base):
    __tablename__ = 'sessions'
    
    id = Column(String, primary_key=True)
    user_id = Column(String)
    device_info = Column(JSON)
    ip_address = Column(String)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class APIKeyDB(Base):
    __tablename__ = 'api_keys'
    
    id = Column(String, primary_key=True)
    user_id = Column(String)
    key_hash = Column(String)
    name = Column(String)
    permissions = Column(JSON)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ServiceTokenDB(Base):
    __tablename__ = 'service_tokens'
    
    id = Column(String, primary_key=True)
    service_name = Column(String)
    token_hash = Column(String)
    permissions = Column(JSON)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    username: str
    password: str
    mfa_code: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    name: str
    permissions: List[Permission]
    expires_in_days: Optional[int] = 365


class AuthService:
    """Production authentication service with comprehensive security features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Crypto setup
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = config['jwt_secret']
        self.jwt_algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=config.get('access_token_expire_minutes', 15))
        self.refresh_token_expire = timedelta(days=config.get('refresh_token_expire_days', 30))
        
        # Encryption for sensitive data
        self.fernet = Fernet(config['encryption_key'].encode())
        
        # Database
        self.engine = create_engine(config['postgres_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis for sessions and rate limiting
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # OAuth2 providers
        self.oauth2_providers = config.get('oauth2_providers', {})
        
        # Security settings
        self.max_failed_attempts = config.get('max_failed_attempts', 5)
        self.lockout_duration = timedelta(minutes=config.get('lockout_minutes', 30))
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def create_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Create new user with secure password hashing"""
        auth_attempts.labels(method='registration', status='attempted').inc()
        
        try:
            # Check if user exists
            with self.Session() as session:
                existing_user = session.query(UserDB).filter(
                    (UserDB.email == user_data.email) | 
                    (UserDB.username == user_data.username)
                ).first()
                
                if existing_user:
                    auth_attempts.labels(method='registration', status='failed').inc()
                    raise HTTPException(status_code=400, detail="User already exists")
                
                # Hash password
                password_hash = self.pwd_context.hash(user_data.password)
                
                # Create user
                user_id = f"user_{uuid.uuid4().hex}"
                user = UserDB(
                    id=user_id,
                    email=user_data.email,
                    username=user_data.username,
                    password_hash=password_hash,
                    role=user_data.role.value,
                    metadata={
                        'registration_ip': None,  # Set from request
                        'email_verification_sent': datetime.utcnow().isoformat()
                    }
                )
                
                session.add(user)
                session.commit()
                
                # Send verification email (implement separately)
                await self._send_verification_email(user_data.email, user_id)
                
                auth_attempts.labels(method='registration', status='success').inc()
                
                return {
                    'user_id': user_id,
                    'email': user_data.email,
                    'status': 'created',
                    'verification_required': True
                }
                
        except Exception as e:
            auth_attempts.labels(method='registration', status='error').inc()
            logger.error(f"User creation failed: {e}")
            raise HTTPException(status_code=500, detail="User creation failed")
    
    async def authenticate_user(self, login_data: UserLogin, 
                              request: Request) -> Dict[str, Any]:
        """Authenticate user with comprehensive security checks"""
        auth_attempts.labels(method='password', status='attempted').inc()
        
        try:
            with self.Session() as session:
                user = session.query(UserDB).filter(
                    UserDB.username == login_data.username
                ).first()
                
                if not user:
                    auth_attempts.labels(method='password', status='failed').inc()
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Check if account is locked
                if user.locked_until and user.locked_until > datetime.utcnow():
                    auth_attempts.labels(method='password', status='locked').inc()
                    raise HTTPException(
                        status_code=423, 
                        detail=f"Account locked until {user.locked_until}"
                    )
                
                # Verify password
                if not self.pwd_context.verify(login_data.password, user.password_hash):
                    # Increment failed attempts
                    user.failed_attempts += 1
                    
                    if user.failed_attempts >= self.max_failed_attempts:
                        user.locked_until = datetime.utcnow() + self.lockout_duration
                        logger.warning(f"Account {user.username} locked due to failed attempts")
                    
                    session.commit()
                    auth_attempts.labels(method='password', status='failed').inc()
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Check if MFA is required
                if user.mfa_enabled:
                    if not login_data.mfa_code:
                        auth_attempts.labels(method='password', status='mfa_required').inc()
                        return {
                            'status': 'mfa_required',
                            'user_id': user.id,
                            'mfa_methods': ['totp']
                        }
                    
                    # Verify MFA code
                    if not self._verify_mfa_code(user.mfa_secret, login_data.mfa_code):
                        auth_attempts.labels(method='mfa', status='failed').inc()
                        raise HTTPException(status_code=401, detail="Invalid MFA code")
                
                # Reset failed attempts on successful login
                user.failed_attempts = 0
                user.locked_until = None
                user.last_login = datetime.utcnow()
                session.commit()
                
                # Create session
                session_data = await self._create_session(
                    user.id, 
                    request.client.host if request.client else 'unknown',
                    request.headers.get('user-agent', 'unknown')
                )
                
                # Generate tokens
                tokens = self._generate_tokens(user.id, user.role, session_data['session_id'])
                
                auth_attempts.labels(method='password', status='success').inc()
                active_sessions.inc()
                
                return {
                    'user_id': user.id,
                    'role': user.role,
                    'tokens': tokens,
                    'session_id': session_data['session_id']
                }
                
        except HTTPException:
            raise
        except Exception as e:
            auth_attempts.labels(method='password', status='error').inc()
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")
    
    async def create_api_key(self, user_id: str, api_key_data: APIKeyCreate) -> Dict[str, Any]:
        """Create API key for user"""
        try:
            # Generate secure API key
            api_key = f"nba_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with self.Session() as session:
                # Verify user exists
                user = session.query(UserDB).filter_by(id=user_id).first()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Create API key record
                api_key_id = f"key_{uuid.uuid4().hex}"
                expires_at = datetime.utcnow() + timedelta(days=api_key_data.expires_in_days)
                
                key_record = APIKeyDB(
                    id=api_key_id,
                    user_id=user_id,
                    key_hash=key_hash,
                    name=api_key_data.name,
                    permissions=[asdict(p) for p in api_key_data.permissions],
                    expires_at=expires_at
                )
                
                session.add(key_record)
                session.commit()
                
                token_operations.labels(operation='create', type='api_key').inc()
                
                return {
                    'api_key_id': api_key_id,
                    'api_key': api_key,  # Only returned once
                    'name': api_key_data.name,
                    'expires_at': expires_at.isoformat(),
                    'permissions': api_key_data.permissions
                }
                
        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            raise HTTPException(status_code=500, detail="API key creation failed")
    
    async def create_service_token(self, service_name: str, 
                                 permissions: List[Permission]) -> Dict[str, Any]:
        """Create service-to-service authentication token"""
        try:
            # Generate service token
            token = f"svc_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(40))}"
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            with self.Session() as session:
                token_id = f"svc_{uuid.uuid4().hex}"
                expires_at = datetime.utcnow() + timedelta(days=365)  # Long-lived for services
                
                service_token = ServiceTokenDB(
                    id=token_id,
                    service_name=service_name,
                    token_hash=token_hash,
                    permissions=[asdict(p) for p in permissions],
                    expires_at=expires_at
                )
                
                session.add(service_token)
                session.commit()
                
                token_operations.labels(operation='create', type='service_token').inc()
                
                return {
                    'token_id': token_id,
                    'service_token': token,
                    'service_name': service_name,
                    'expires_at': expires_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Service token creation failed: {e}")
            raise HTTPException(status_code=500, detail="Service token creation failed")
    
    async def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """Setup Multi-Factor Authentication for user"""
        try:
            with self.Session() as session:
                user = session.query(UserDB).filter_by(id=user_id).first()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Generate TOTP secret
                secret = pyotp.random_base32()
                
                # Create QR code
                totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                    name=user.email,
                    issuer_name="NBA Analytics"
                )
                
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(totp_uri)
                qr.make(fit=True)
                
                img = qr.make_image(fill_color="black", back_color="white")
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Store encrypted secret (don't enable MFA yet)
                encrypted_secret = self.fernet.encrypt(secret.encode()).decode()
                user.mfa_secret = encrypted_secret
                session.commit()
                
                return {
                    'secret': secret,
                    'qr_code': f"data:image/png;base64,{qr_code_base64}",
                    'backup_codes': self._generate_backup_codes()
                }
                
        except Exception as e:
            logger.error(f"MFA setup failed: {e}")
            raise HTTPException(status_code=500, detail="MFA setup failed")
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            user_id = payload.get('sub')
            token_type = payload.get('type')
            session_id = payload.get('session_id')
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Check if session is still active
            if session_id:
                session_key = f"session:{session_id}"
                if not self.redis_client.exists(session_key):
                    raise HTTPException(status_code=401, detail="Session expired")
            
            # Get user info
            with self.Session() as session:
                user = session.query(UserDB).filter_by(id=user_id).first()
                if not user or not user.is_active:
                    raise HTTPException(status_code=401, detail="User not found or inactive")
            
            token_operations.labels(operation='verify', type=token_type).inc()
            
            return {
                'user_id': user_id,
                'role': user.role,
                'token_type': token_type,
                'session_id': session_id
            }
            
        except jwt.ExpiredSignatureError:
            token_operations.labels(operation='verify', type='expired').inc()
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            token_operations.labels(operation='verify', type='invalid').inc()
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    async def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with self.Session() as session:
                key_record = session.query(APIKeyDB).filter_by(
                    key_hash=key_hash,
                    is_active=True
                ).first()
                
                if not key_record:
                    auth_attempts.labels(method='api_key', status='failed').inc()
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                # Check expiration
                if key_record.expires_at < datetime.utcnow():
                    auth_attempts.labels(method='api_key', status='expired').inc()
                    raise HTTPException(status_code=401, detail="API key expired")
                
                # Update last used
                key_record.last_used = datetime.utcnow()
                session.commit()
                
                # Get user info
                user = session.query(UserDB).filter_by(id=key_record.user_id).first()
                
                auth_attempts.labels(method='api_key', status='success').inc()
                
                return {
                    'user_id': key_record.user_id,
                    'role': user.role if user else 'api_user',
                    'permissions': key_record.permissions,
                    'api_key_id': key_record.id
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            raise HTTPException(status_code=401, detail="API key verification failed")
    
    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        try:
            with self.Session() as session:
                user = session.query(UserDB).filter_by(id=user_id).first()
                
                if not user:
                    permission_checks.labels(resource=permission.resource, result='denied').inc()
                    return False
                
                # Admin has all permissions
                if user.role == UserRole.ADMIN.value:
                    permission_checks.labels(resource=permission.resource, result='granted').inc()
                    return True
                
                # Check role-based permissions
                role_permissions = self._get_role_permissions(user.role)
                
                for role_perm in role_permissions:
                    if (role_perm['resource'] == permission.resource and 
                        role_perm['action'] == permission.action):
                        
                        # Check scope if specified
                        if permission.scope and role_perm.get('scope'):
                            if permission.scope != role_perm['scope']:
                                continue
                        
                        permission_checks.labels(resource=permission.resource, result='granted').inc()
                        return True
                
                permission_checks.labels(resource=permission.resource, result='denied').inc()
                return False
                
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            permission_checks.labels(resource=permission.resource, result='error').inc()
            return False
    
    async def logout(self, session_id: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            # Remove session from Redis
            session_key = f"session:{session_id}"
            self.redis_client.delete(session_key)
            
            # Update database
            with self.Session() as session:
                session_record = session.query(SessionDB).filter_by(id=session_id).first()
                if session_record:
                    session_record.is_active = False
                    session.commit()
            
            active_sessions.dec()
            
            return {'status': 'logged_out'}
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            raise HTTPException(status_code=500, detail="Logout failed")
    
    def _generate_tokens(self, user_id: str, role: str, session_id: str) -> TokenResponse:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token
        access_payload = {
            'sub': user_id,
            'role': role,
            'type': 'access',
            'session_id': session_id,
            'iat': now,
            'exp': now + self.access_token_expire
        }
        
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Refresh token
        refresh_payload = {
            'sub': user_id,
            'type': 'refresh',
            'session_id': session_id,
            'iat': now,
            'exp': now + self.refresh_token_expire
        }
        
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        token_operations.labels(operation='create', type='access').inc()
        token_operations.labels(operation='create', type='refresh').inc()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds())
        )
    
    async def _create_session(self, user_id: str, ip_address: str, 
                            user_agent: str) -> Dict[str, Any]:
        """Create user session"""
        session_id = f"sess_{uuid.uuid4().hex}"
        expires_at = datetime.utcnow() + self.refresh_token_expire
        
        # Store in Redis for fast access
        session_data = {
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': expires_at.isoformat()
        }
        
        session_key = f"session:{session_id}"
        self.redis_client.setex(
            session_key,
            int(self.refresh_token_expire.total_seconds()),
            json.dumps(session_data)
        )
        
        # Store in database for persistence
        with self.Session() as session:
            session_record = SessionDB(
                id=session_id,
                user_id=user_id,
                device_info={'user_agent': user_agent},
                ip_address=ip_address,
                expires_at=expires_at
            )
            session.add(session_record)
            session.commit()
        
        return {'session_id': session_id}
    
    def _verify_mfa_code(self, encrypted_secret: str, code: str) -> bool:
        """Verify TOTP MFA code"""
        try:
            secret = self.fernet.decrypt(encrypted_secret.encode()).decode()
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except Exception:
            return False
    
    def _get_role_permissions(self, role: str) -> List[Dict[str, str]]:
        """Get permissions for role"""
        permissions = {
            UserRole.ADMIN.value: [
                {'resource': '*', 'action': '*'},
            ],
            UserRole.USER.value: [
                {'resource': 'predictions', 'action': 'read'},
                {'resource': 'predictions', 'action': 'create'},
                {'resource': 'players', 'action': 'read'},
                {'resource': 'teams', 'action': 'read'},
            ],
            UserRole.READONLY.value: [
                {'resource': 'predictions', 'action': 'read'},
                {'resource': 'players', 'action': 'read'},
                {'resource': 'teams', 'action': 'read'},
            ]
        }
        
        return permissions.get(role, [])
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate MFA backup codes"""
        return [secrets.token_hex(4).upper() for _ in range(10)]
    
    async def _send_verification_email(self, email: str, user_id: str):
        """Send email verification (implement with your email service)"""
        # Implement email sending logic
        pass


# FastAPI dependencies
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

auth_service = None

async def get_auth_service():
    global auth_service
    if not auth_service:
        config = {
            'jwt_secret': 'your-secret-key',
            'encryption_key': Fernet.generate_key().decode(),
            'postgres_url': 'postgresql://user:pass@localhost/auth',
            'redis_host': 'localhost',
            'redis_port': 6379
        }
        auth_service = AuthService(config)
    return auth_service

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_svc: AuthService = Depends(get_auth_service)
):
    """Get current authenticated user"""
    token = credentials.credentials
    return await auth_svc.verify_token(token)

async def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    def permission_dependency(
        current_user: Dict = Depends(get_current_user),
        auth_svc: AuthService = Depends(get_auth_service)
    ):
        has_permission = asyncio.run(
            auth_svc.check_permission(current_user['user_id'], permission)
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail=f"Permission required: {permission.action} on {permission.resource}"
            )
        
        return current_user
    
    return permission_dependency


# FastAPI app
app = FastAPI(title="NBA Analytics Authentication Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register")
async def register(
    user_data: UserCreate,
    auth_svc: AuthService = Depends(get_auth_service)
):
    """Register new user"""
    return await auth_svc.create_user(user_data)

@app.post("/login")
async def login(
    login_data: UserLogin,
    request: Request,
    auth_svc: AuthService = Depends(get_auth_service)
):
    """User login"""
    return await auth_svc.authenticate_user(login_data, request)

@app.post("/api-keys")
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: Dict = Depends(get_current_user),
    auth_svc: AuthService = Depends(get_auth_service)
):
    """Create API key"""
    return await auth_svc.create_api_key(current_user['user_id'], api_key_data)

@app.get("/me")
async def get_me(current_user: Dict = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@app.post("/logout")
async def logout(
    current_user: Dict = Depends(get_current_user),
    auth_svc: AuthService = Depends(get_auth_service)
):
    """Logout user"""
    return await auth_svc.logout(current_user['session_id'])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)