"""Проверка адресов, которые вводит пользователь: защита от обращения сервера к внутренней сети."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class UnsafeUrl(ValueError):
    pass


def _resolve(host: str) -> list:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _addr_ok(text: str) -> bool:
    try:
        return ipaddress.ip_address(text.split("%")[0]).is_global
    except ValueError:
        return False


def check_llm_url(url: str, resolve: bool = False) -> str:
    """Адрес модели или агента: https, либо http только для сервиса на этом же компьютере (localhost).
    resolve=True дополнительно проверяет, что имя узла не указывает во внутреннюю сеть."""
    u = urlparse(str(url or "").strip())
    if u.scheme not in ("https", "http") or not u.hostname:
        raise UnsafeUrl("адрес должен начинаться с https:// (для сервиса на этом компьютере допустим http://localhost)")
    if u.username or u.password:
        raise UnsafeUrl("логин и пароль в адресе запрещены: ключи вводятся в отдельном поле")
    host = u.hostname.lower()
    if u.scheme == "http" and host not in LOOPBACK:
        raise UnsafeUrl("http разрешён только для localhost")
    if host in LOOPBACK:
        return url.strip().rstrip("/")
    if host.endswith((".local", ".internal", ".localhost")):
        raise UnsafeUrl("внутренние адреса запрещены")
    try:
        ipaddress.ip_address(host)
        literal = True
    except ValueError:
        literal = False
    if literal and not _addr_ok(host):
        raise UnsafeUrl("адрес из частной или служебной сети запрещён")
    if resolve and not literal:
        try:
            addrs = _resolve(host)
        except OSError as exc:
            raise UnsafeUrl(f"не удалось определить адрес узла: {exc}")
        if not addrs or not all(_addr_ok(a) for a in addrs):
            raise UnsafeUrl("узел указывает на частную или служебную сеть")
    return url.strip().rstrip("/")
