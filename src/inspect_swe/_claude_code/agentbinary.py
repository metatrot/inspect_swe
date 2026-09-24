import re
from pathlib import Path

from inspect_ai.util import SandboxEnvironment
from pydantic import BaseModel
from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform, sandbox_exec

SYSTEM_PROMPT_FILE_MIN_VERSION = (2, 0, 33)


def claude_code_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("claude-code-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        base_url = await _claude_code_download_base_url()
        version = await _claude_code_version(base_url, version)
        manifest = await _claude_code_manifest(base_url, version)
        expected_checksum = _checksum_for_platform(manifest, platform)
        download_url = f"{base_url}/{version}/{platform}/claude"
        return AgentBinaryVersion(version, expected_checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"claude-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("claude-*"))

    return AgentBinarySource(
        agent="claude code",
        binary="claude",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
    )


async def claude_code_binary_version(
    sandbox: SandboxEnvironment, binary: str, user: str | None = None
) -> tuple[int, int, int]:
    output = await sandbox_exec(sandbox, f"{binary} --version", user=user)
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(f"Unable to parse claude code version from {output!r}")
    return int(match[1]), int(match[2]), int(match[3])


async def claude_code_supports_system_prompt_files(
    sandbox: SandboxEnvironment, binary: str, user: str | None = None
) -> bool:
    version = await claude_code_binary_version(sandbox, binary, user)
    return version >= SYSTEM_PROMPT_FILE_MIN_VERSION


async def _claude_code_download_base_url() -> str:
    INSTALL_SCRIPT_URL = "https://claude.ai/install.sh"
    script_content = await download_text_file(INSTALL_SCRIPT_URL)
    for pattern in [
        r'DOWNLOAD_BASE_URL="(https://[^"]+)"',
        r'GCS_BUCKET="(https://[^"]+)"',
    ]:
        match = re.search(pattern, script_content)
        if match is not None:
            return match.group(1)
    raise RuntimeError("Unable to determine download base URL for claude code.")


async def _claude_code_version(base_url: str, target: str) -> str:
    # validate target
    target_pattern = r"^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$"
    if re.match(target_pattern, target) is None:
        raise RuntimeError(
            "Invalid version target (must be 'stable', 'latest', or a semver version number)"
        )

    # resolve target alias if required
    if target in ["stable", "latest"]:
        version_url = f"{base_url}/{target}"
        version = await download_text_file(version_url)
        return version
    else:
        return target


class PlatformInfo(BaseModel):
    checksum: str
    size: int


class Manifest(BaseModel):
    version: str
    platforms: dict[str, PlatformInfo]


async def _claude_code_manifest(base_url: str, version: str) -> Manifest:
    manifest_url = f"{base_url}/{version}/manifest.json"
    manifest_json = await download_text_file(manifest_url)
    return Manifest.model_validate_json(manifest_json)


def _checksum_for_platform(manifest: Manifest, platform: SandboxPlatform) -> str:
    if platform not in manifest.platforms:
        raise RuntimeError(f"Platform '{platform}' not found in manifest.")
    return manifest.platforms[platform].checksum
