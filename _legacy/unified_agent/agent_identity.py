#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 에이전트 ID 모듈 (Agent Identity Module)

================================================================================
📁 파일 위치: unified_agent/agent_identity.py
📋 역할: Microsoft Entra ID 기반 에이전트 인증/인가, 에이전트 ID 관리
📅 최종 업데이트: 2026년 2월 13일
📦 버전: v4.1.0
✅ 테스트: test_v41_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. AgentIdentity - 에이전트 전용 ID (Microsoft Entra Agent Identity)
    2. AgentCredential - 에이전트 인증 토큰 및 자격 증명 관리
    3. AgentRBACManager - Azure RBAC 기반 에이전트 권한 관리
    4. AgentIdentityProvider - 에이전트 ID 프로비저닝 및 생명주기 관리
    5. ScopedPermission - 에이전트별 도구/리소스 접근 제어

🔧 2026년 2월 기능:
    - Microsoft Entra ID에서 에이전트 전용 ID 타입 지원
    - 에이전트별 최소 권한 원칙 (Least Privilege) 적용
    - Azure RBAC를 통한 에이전트 리소스 접근 제어
    - 에이전트 ID 생명주기 자동 관리 (프로비저닝 → 해제)
    - 에이전트 간 위임(Delegation) 인증
    - 감사 로그 (Audit Trail) 통합

📌 사용 예시:
    >>> from unified_agent.agent_identity import (
    ...     AgentIdentity, AgentCredential, AgentRBACManager,
    ...     AgentIdentityProvider, ScopedPermission, PermissionScope
    ... )
    >>>
    >>> # 에이전트 ID 생성 및 인증
    >>> provider = AgentIdentityProvider(tenant_id="your-tenant-id")
    >>> identity = await provider.provision_agent(
    ...     name="research-agent",
    ...     scopes=[PermissionScope.SEARCH, PermissionScope.FILE_READ]
    ... )
    >>>
    >>> # RBAC로 에이전트 접근 제어
    >>> rbac = AgentRBACManager()
    >>> rbac.assign_role(identity.agent_id, "AI.Agent.Reader")
    >>> rbac.check_permission(identity.agent_id, "storage:read")  # True

⚠️ 주의사항:
    - 프로덕션에서는 Azure Entra ID 실제 통합이 필요합니다.
    - 에이전트별 최소 권한 원칙을 반드시 적용하세요.
    - 에이전트 자격 증명은 정기적으로 갱신해야 합니다.

🔗 관련 문서:
    - Agent Identity in Foundry: https://learn.microsoft.com/azure/ai-foundry/agents/concepts/agent-identity
    - Azure RBAC: https://learn.microsoft.com/azure/role-based-access-control/overview
    - Microsoft Entra ID: https://learn.microsoft.com/entra/fundamentals/what-is-entra
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, unique
from typing import Any

__all__ = [
    # Enums
    "PermissionScope",
    "IdentityStatus",
    "AuthMethod",
    "AgentRole",
    # Data Models
    "AgentIdentity",
    "AgentCredential",
    "ScopedPermission",
    "IdentityAuditEntry",
    # Core Components
    "AgentRBACManager",
    "AgentIdentityProvider",
    "AgentDelegation",
    "IdentityRegistry",
]

logger = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================

@unique
class PermissionScope(Enum):
    """
    에이전트 권한 범위 (Permission Scope)

    에이전트가 접근할 수 있는 리소스 및 작업의 범위를 정의합니다.
    Azure RBAC의 세분화된 권한 모델을 반영합니다.
    """
    # 데이터 접근
    FILE_READ = "file:read"           # 파일 읽기
    FILE_WRITE = "file:write"         # 파일 쓰기
    STORAGE_READ = "storage:read"     # 스토리지 읽기
    STORAGE_WRITE = "storage:write"   # 스토리지 쓰기

    # AI 서비스
    MODEL_INVOKE = "model:invoke"     # 모델 호출
    SEARCH = "search:query"           # 검색 실행
    EMBEDDING = "embedding:create"    # 임베딩 생성
    CODE_EXECUTE = "code:execute"     # 코드 실행

    # 에이전트 간 통신
    AGENT_DELEGATE = "agent:delegate" # 다른 에이전트에 위임
    AGENT_DISCOVER = "agent:discover" # 에이전트 검색 (A2A)
    MCP_CONNECT = "mcp:connect"       # MCP 서버 연결

    # 외부 서비스
    WEB_ACCESS = "web:access"         # 웹 접근 (Bing Search 등)
    API_CALL = "api:call"             # 외부 API 호출
    BROWSER_USE = "browser:use"       # 브라우저 자동화

    # 관리
    ADMIN = "admin:all"               # 전체 관리자 권한


@unique
class IdentityStatus(Enum):
    """에이전트 ID 상태"""
    PROVISIONING = "provisioning"     # 프로비저닝 중
    ACTIVE = "active"                 # 활성
    SUSPENDED = "suspended"           # 일시 중지
    EXPIRED = "expired"               # 만료
    REVOKED = "revoked"               # 해제/취소


@unique
class AuthMethod(Enum):
    """에이전트 인증 방법"""
    MANAGED_IDENTITY = "managed_identity"  # Azure Managed Identity
    CLIENT_SECRET = "client_secret"        # Client ID + Secret
    CERTIFICATE = "certificate"            # 인증서 기반
    FEDERATED = "federated"                # Federated Identity
    TOKEN = "token"                        # Bearer Token


@unique
class AgentRole(Enum):
    """
    사전 정의된 에이전트 역할 (Azure RBAC 패턴)

    각 역할은 미리 정의된 권한 세트를 가집니다.
    """
    READER = "AI.Agent.Reader"             # 읽기 전용 (모델 호출, 검색)
    CONTRIBUTOR = "AI.Agent.Contributor"    # 읽기/쓰기 (파일, 스토리지 포함)
    OPERATOR = "AI.Agent.Operator"         # 실행 + 위임 (에이전트 간 통신)
    ADMIN = "AI.Agent.Admin"               # 전체 관리 권한

    @property
    def default_scopes(self) -> set[PermissionScope]:
        """역할별 기본 권한 범위 반환"""
        _role_scopes = {
            AgentRole.READER: {
                PermissionScope.FILE_READ,
                PermissionScope.STORAGE_READ,
                PermissionScope.MODEL_INVOKE,
                PermissionScope.SEARCH,
                PermissionScope.EMBEDDING,
            },
            AgentRole.CONTRIBUTOR: {
                PermissionScope.FILE_READ, PermissionScope.FILE_WRITE,
                PermissionScope.STORAGE_READ, PermissionScope.STORAGE_WRITE,
                PermissionScope.MODEL_INVOKE, PermissionScope.SEARCH,
                PermissionScope.EMBEDDING, PermissionScope.CODE_EXECUTE,
            },
            AgentRole.OPERATOR: {
                PermissionScope.FILE_READ, PermissionScope.FILE_WRITE,
                PermissionScope.STORAGE_READ, PermissionScope.STORAGE_WRITE,
                PermissionScope.MODEL_INVOKE, PermissionScope.SEARCH,
                PermissionScope.EMBEDDING, PermissionScope.CODE_EXECUTE,
                PermissionScope.AGENT_DELEGATE, PermissionScope.AGENT_DISCOVER,
                PermissionScope.MCP_CONNECT, PermissionScope.WEB_ACCESS,
                PermissionScope.API_CALL,
            },
            AgentRole.ADMIN: set(PermissionScope),
        }
        return _role_scopes.get(self, set())


# ============================================================================
# Data Models
# ============================================================================

@dataclass(slots=True)
class AgentCredential:
    """
    에이전트 자격 증명 (Agent Credential)

    에이전트가 Azure 리소스에 접근하기 위한 인증 정보를 관리합니다.
    자격 증명은 자동 갱신 및 만료 관리가 가능합니다.

    Attributes:
        credential_id: 자격 증명 고유 ID
        auth_method: 인증 방법 (Managed Identity, Token 등)
        token: 현재 유효한 액세스 토큰
        expires_at: 토큰 만료 시각
        refresh_token: 갱신용 토큰 (옵션)
        tenant_id: Azure AD 테넌트 ID
    """
    credential_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    auth_method: AuthMethod = AuthMethod.MANAGED_IDENTITY
    token: str = ""
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    refresh_token: str | None = None
    tenant_id: str = ""
    _created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_expired(self) -> bool:
        """토큰 만료 여부 확인"""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def remaining_seconds(self) -> float:
        """남은 유효 시간 (초 단위)"""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds())

    def refresh(self, new_token: str, ttl_seconds: int = 3600) -> None:
        """토큰 갱신"""
        self.token = new_token
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        logger.debug(f"Credential {self.credential_id[:8]}... refreshed, TTL={ttl_seconds}s")


@dataclass(slots=True)
class ScopedPermission:
    """
    범위 지정 권한 (Scoped Permission)

    특정 리소스에 대한 에이전트의 세분화된 권한을 정의합니다.

    Attributes:
        scope: 권한 범위 (PermissionScope)
        resource_pattern: 리소스 패턴 (예: "storage/container-*")
        conditions: 추가 조건 (예: {"time_range": "09:00-18:00"})
        granted_at: 권한 부여 시각
        expires_at: 권한 만료 시각 (None이면 무기한)
    """
    scope: PermissionScope
    resource_pattern: str = "*"
    conditions: dict[str, Any] = field(default_factory=dict)
    granted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def matches_resource(self, resource: str) -> bool:
        """리소스 패턴 매칭 확인"""
        if self.resource_pattern == "*":
            return True
        # 간단한 와일드카드 패턴 매칭
        return fnmatch.fnmatch(resource, self.resource_pattern)


@dataclass(slots=True)
class AgentIdentity:
    """
    에이전트 ID (Agent Identity)

    Microsoft Entra ID의 에이전트 전용 ID를 나타냅니다.
    에이전트의 인증, 인가, 감사를 위한 핵심 엔티티입니다.

    Attributes:
        agent_id: 에이전트 고유 ID
        name: 에이전트 표시 이름
        description: 에이전트 설명
        status: 에이전트 ID 상태
        role: 에이전트 역할
        credential: 에이전트 자격 증명
        permissions: 세분화된 권한 목록
        metadata: 추가 메타데이터
        created_at: 생성 시각
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: IdentityStatus = IdentityStatus.PROVISIONING
    role: AgentRole = AgentRole.READER
    credential: AgentCredential | None = None
    permissions: list[ScopedPermission] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    parent_agent_id: str | None = None    # 위임 시 부모 에이전트 ID

    @property
    def is_active(self) -> bool:
        return self.status == IdentityStatus.ACTIVE

    @property
    def effective_scopes(self) -> set[PermissionScope]:
        """역할 기본 권한 + 개별 부여 권한의 합집합"""
        scopes = self.role.default_scopes.copy()
        for perm in self.permissions:
            if not perm.is_expired:
                scopes.add(perm.scope)
        return scopes

    def has_permission(self, scope: PermissionScope, resource: str = "*") -> bool:
        """특정 권한 보유 여부 확인"""
        if not self.is_active:
            return False
        # Admin은 모든 권한
        if PermissionScope.ADMIN in self.effective_scopes:
            return True
        # 역할 기본 권한 확인
        if scope in self.role.default_scopes:
            return True
        # 개별 권한 확인
        for perm in self.permissions:
            if perm.scope == scope and not perm.is_expired:
                if perm.matches_resource(resource):
                    return True
        return False


@dataclass(slots=True)
class IdentityAuditEntry:
    """
    에이전트 ID 감사 로그 항목

    에이전트 ID 관련 모든 작업을 기록합니다.

    Attributes:
        entry_id: 로그 항목 ID
        agent_id: 대상 에이전트 ID
        action: 수행된 작업 (provision, revoke, check_permission 등)
        result: 작업 결과
        details: 상세 정보
        timestamp: 발생 시각
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    action: str = ""
    result: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ============================================================================
# Core Components
# ============================================================================

class AgentRBACManager:
    """
    에이전트 RBAC 관리자 (Role-Based Access Control Manager)

    Azure RBAC 패턴을 따르는 에이전트별 접근 제어를 관리합니다.
    역할 할당, 권한 확인, 감사 로그를 제공합니다.

    📌 사용 예시:
        >>> rbac = AgentRBACManager()
        >>> rbac.assign_role("agent-123", AgentRole.CONTRIBUTOR)
        >>> rbac.check_permission("agent-123", PermissionScope.FILE_WRITE)
        True
        >>> rbac.grant_permission("agent-123", ScopedPermission(
        ...     scope=PermissionScope.WEB_ACCESS,
        ...     resource_pattern="bing.com/*"
        ... ))
    """

    def __init__(self) -> None:
        self._identities: dict[str, AgentIdentity] = {}
        self._audit_log: list[IdentityAuditEntry] = []
        logger.info("AgentRBACManager initialized")

    def register_identity(self, identity: AgentIdentity) -> None:
        """에이전트 ID 등록"""
        self._identities[identity.agent_id] = identity
        self._log_audit(identity.agent_id, "register", "success")

    def assign_role(self, agent_id: str, role: AgentRole) -> bool:
        """역할 할당"""
        identity = self._identities.get(agent_id)
        if not identity:
            self._log_audit(agent_id, "assign_role", "failed", {"reason": "not_found"})
            return False
        identity.role = role
        self._log_audit(agent_id, "assign_role", "success", {"role": role.value})
        logger.info(f"Agent {agent_id[:8]}... assigned role: {role.value}")
        return True

    def grant_permission(self, agent_id: str, permission: ScopedPermission) -> bool:
        """개별 권한 부여"""
        identity = self._identities.get(agent_id)
        if not identity:
            return False
        identity.permissions.append(permission)
        self._log_audit(agent_id, "grant_permission", "success", {
            "scope": permission.scope.value,
            "resource": permission.resource_pattern,
        })
        return True

    def revoke_permission(self, agent_id: str, scope: PermissionScope) -> bool:
        """특정 권한 해제"""
        identity = self._identities.get(agent_id)
        if not identity:
            return False
        identity.permissions = [
            p for p in identity.permissions if p.scope != scope
        ]
        self._log_audit(agent_id, "revoke_permission", "success", {
            "scope": scope.value,
        })
        return True

    def check_permission(
        self, agent_id: str, scope: PermissionScope, resource: str = "*"
    ) -> bool:
        """권한 확인"""
        identity = self._identities.get(agent_id)
        if not identity:
            self._log_audit(agent_id, "check_permission", "denied", {
                "reason": "identity_not_found"
            })
            return False
        result = identity.has_permission(scope, resource)
        self._log_audit(agent_id, "check_permission",
                        "allowed" if result else "denied", {
                            "scope": scope.value,
                            "resource": resource,
                        })
        return result

    def get_identity(self, agent_id: str) -> AgentIdentity | None:
        """에이전트 ID 조회"""
        return self._identities.get(agent_id)

    def list_identities(self, status: IdentityStatus | None = None) -> list[AgentIdentity]:
        """에이전트 ID 목록 조회 (상태별 필터링 가능)"""
        if status is None:
            return list(self._identities.values())
        return [i for i in self._identities.values() if i.status == status]

    def get_audit_log(
        self, agent_id: str | None = None, limit: int = 100
    ) -> list[IdentityAuditEntry]:
        """감사 로그 조회"""
        logs = self._audit_log
        if agent_id:
            logs = [e for e in logs if e.agent_id == agent_id]
        return logs[-limit:]

    def _log_audit(
        self, agent_id: str, action: str, result: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """감사 로그 기록"""
        entry = IdentityAuditEntry(
            agent_id=agent_id,
            action=action,
            result=result,
            details=details or {},
        )
        self._audit_log.append(entry)


class AgentDelegation:
    """
    에이전트 위임 관리 (Agent Delegation)

    에이전트가 다른 에이전트에게 권한을 위임하는 체인을 관리합니다.
    위임 시 부모 에이전트의 권한 범위를 초과할 수 없습니다.

    📌 사용 예시:
        >>> delegation = AgentDelegation(rbac_manager)
        >>> child = delegation.delegate(
        ...     parent_agent_id="parent-agent",
        ...     child_name="sub-task-agent",
        ...     delegated_scopes=[PermissionScope.SEARCH]
        ... )
    """

    def __init__(self, rbac: AgentRBACManager) -> None:
        self._rbac = rbac
        self._delegations: dict[str, list[str]] = {}  # parent → [child IDs]

    def delegate(
        self,
        parent_agent_id: str,
        child_name: str,
        delegated_scopes: list[PermissionScope],
    ) -> AgentIdentity | None:
        """부모 에이전트가 자식 에이전트에게 권한 위임"""
        parent = self._rbac.get_identity(parent_agent_id)
        if not parent or not parent.is_active:
            logger.warning(f"Delegation failed: parent {parent_agent_id[:8]}... not active")
            return None

        # 부모 권한 범위를 초과하는 위임 방지
        parent_scopes = parent.effective_scopes
        invalid_scopes = [s for s in delegated_scopes if s not in parent_scopes]
        if invalid_scopes:
            logger.warning(
                f"Delegation failed: scopes {invalid_scopes} exceed parent permissions"
            )
            return None

        # 자식 에이전트 생성
        child = AgentIdentity(
            name=child_name,
            description=f"Delegated from {parent.name}",
            status=IdentityStatus.ACTIVE,
            role=AgentRole.READER,  # 최소 역할
            parent_agent_id=parent_agent_id,
            credential=AgentCredential(
                auth_method=AuthMethod.TOKEN,
                token=hashlib.sha256(
                    f"{parent_agent_id}:{child_name}:{time.time()}".encode()
                ).hexdigest(),
            ),
            permissions=[
                ScopedPermission(scope=scope) for scope in delegated_scopes
            ],
        )

        self._rbac.register_identity(child)
        if parent_agent_id not in self._delegations:
            self._delegations[parent_agent_id] = []
        self._delegations[parent_agent_id].append(child.agent_id)

        logger.info(
            f"Agent {parent.name} delegated {len(delegated_scopes)} scopes to {child_name}"
        )
        return child

    def revoke_delegation(self, parent_agent_id: str, child_agent_id: str) -> bool:
        """위임 해제"""
        child = self._rbac.get_identity(child_agent_id)
        if not child or child.parent_agent_id != parent_agent_id:
            return False
        child.status = IdentityStatus.REVOKED
        if parent_agent_id in self._delegations:
            self._delegations[parent_agent_id] = [
                cid for cid in self._delegations[parent_agent_id]
                if cid != child_agent_id
            ]
        return True

    def get_delegation_chain(self, agent_id: str) -> list[str]:
        """에이전트의 위임 체인 조회 (자식 에이전트 목록)"""
        return self._delegations.get(agent_id, [])


class AgentIdentityProvider:
    """
    에이전트 ID 프로바이더 (Agent Identity Provider)

    에이전트 ID의 생명주기(프로비저닝 → 활성 → 갱신 → 해제)를 관리합니다.
    Microsoft Foundry에서 에이전트 ID를 자동으로 프로비저닝하는 것을 시뮬레이션합니다.

    📌 사용 예시:
        >>> provider = AgentIdentityProvider(tenant_id="my-tenant")
        >>> identity = await provider.provision_agent(
        ...     name="research-agent",
        ...     role=AgentRole.OPERATOR,
        ...     scopes=[PermissionScope.SEARCH, PermissionScope.WEB_ACCESS]
        ... )
        >>> print(f"Agent ID: {identity.agent_id}")
        >>> print(f"Status: {identity.status}")
    """

    def __init__(self, tenant_id: str = "default-tenant") -> None:
        self._tenant_id = tenant_id
        self._rbac = AgentRBACManager()
        self._delegation = AgentDelegation(self._rbac)

    @property
    def rbac(self) -> AgentRBACManager:
        return self._rbac

    @property
    def delegation(self) -> AgentDelegation:
        return self._delegation

    async def provision_agent(
        self,
        name: str,
        role: AgentRole = AgentRole.READER,
        scopes: list[PermissionScope] | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        auth_method: AuthMethod = AuthMethod.MANAGED_IDENTITY,
    ) -> AgentIdentity:
        """
        에이전트 ID 프로비저닝 (생성 및 활성화)

        Args:
            name: 에이전트 이름
            role: 에이전트 역할
            scopes: 추가 권한 범위 (역할 기본 권한 외)
            description: 에이전트 설명
            metadata: 추가 메타데이터
            auth_method: 인증 방법

        Returns:
            활성화된 AgentIdentity 객체
        """
        # 1. 자격 증명 생성
        credential = AgentCredential(
            auth_method=auth_method,
            token=hashlib.sha256(
                f"{self._tenant_id}:{name}:{time.time()}".encode()
            ).hexdigest(),
            tenant_id=self._tenant_id,
        )

        # 2. 에이전트 ID 생성
        permissions = [
            ScopedPermission(scope=scope) for scope in (scopes or [])
        ]
        identity = AgentIdentity(
            name=name,
            description=description or f"Agent: {name}",
            status=IdentityStatus.ACTIVE,
            role=role,
            credential=credential,
            permissions=permissions,
            metadata=metadata or {},
        )

        # 3. RBAC에 등록
        self._rbac.register_identity(identity)

        logger.info(
            f"Agent '{name}' provisioned: id={identity.agent_id[:8]}..., "
            f"role={role.value}, scopes={len(identity.effective_scopes)}"
        )
        return identity

    async def suspend_agent(self, agent_id: str) -> bool:
        """에이전트 ID 일시 중지"""
        identity = self._rbac.get_identity(agent_id)
        if not identity:
            return False
        identity.status = IdentityStatus.SUSPENDED
        logger.info(f"Agent {agent_id[:8]}... suspended")
        return True

    async def activate_agent(self, agent_id: str) -> bool:
        """에이전트 ID 활성화"""
        identity = self._rbac.get_identity(agent_id)
        if not identity:
            return False
        identity.status = IdentityStatus.ACTIVE
        logger.info(f"Agent {agent_id[:8]}... activated")
        return True

    async def revoke_agent(self, agent_id: str) -> bool:
        """에이전트 ID 해제 (영구 비활성화)"""
        identity = self._rbac.get_identity(agent_id)
        if not identity:
            return False
        identity.status = IdentityStatus.REVOKED
        if identity.credential:
            identity.credential.token = ""
        logger.info(f"Agent {agent_id[:8]}... revoked")
        return True

    async def refresh_credential(
        self, agent_id: str, ttl_seconds: int = 3600
    ) -> bool:
        """에이전트 자격 증명 갱신"""
        identity = self._rbac.get_identity(agent_id)
        if not identity or not identity.credential:
            return False
        new_token = hashlib.sha256(
            f"{agent_id}:{time.time()}".encode()
        ).hexdigest()
        identity.credential.refresh(new_token, ttl_seconds)
        return True


class IdentityRegistry:
    """
    에이전트 ID 레지스트리 (Identity Registry)

    여러 AgentIdentityProvider를 통합 관리하는 글로벌 레지스트리입니다.
    멀티 테넌트 환경에서 에이전트 ID를 검색하고 관리할 수 있습니다.

    📌 사용 예시:
        >>> registry = IdentityRegistry()
        >>> registry.register_provider("tenant-1", provider1)
        >>> registry.register_provider("tenant-2", provider2)
        >>> identity = registry.find_agent("research-agent")
    """

    def __init__(self) -> None:
        self._providers: dict[str, AgentIdentityProvider] = {}

    def register_provider(self, tenant_id: str, provider: AgentIdentityProvider) -> None:
        """테넌트별 프로바이더 등록"""
        self._providers[tenant_id] = provider

    def find_agent(self, name: str) -> AgentIdentity | None:
        """에이전트 이름으로 전체 테넌트 검색"""
        for provider in self._providers.values():
            for identity in provider.rbac.list_identities():
                if identity.name == name:
                    return identity
        return None

    def find_agent_by_id(self, agent_id: str) -> AgentIdentity | None:
        """에이전트 ID로 전체 테넌트 검색"""
        for provider in self._providers.values():
            identity = provider.rbac.get_identity(agent_id)
            if identity:
                return identity
        return None

    def list_all_agents(
        self, status: IdentityStatus | None = None
    ) -> list[AgentIdentity]:
        """전체 에이전트 목록 (상태별 필터링 가능)"""
        agents = []
        for provider in self._providers.values():
            agents.extend(provider.rbac.list_identities(status))
        return agents

    @property
    def total_agents(self) -> int:
        return sum(
            len(p.rbac.list_identities()) for p in self._providers.values()
        )

    @property
    def active_agents(self) -> int:
        return sum(
            len(p.rbac.list_identities(IdentityStatus.ACTIVE))
            for p in self._providers.values()
        )
