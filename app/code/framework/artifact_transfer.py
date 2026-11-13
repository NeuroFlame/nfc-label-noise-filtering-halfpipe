"""Transfer computation artifacts with NVFlare's supported file streamer."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.streamers.file_streamer import FileStreamer

from .artifacts import ArtifactRef, validate_artifact_name
from .shared import resolve_output_directory

ARTIFACT_COMPONENT_ID = "artifact_transfer"
ARTIFACT_MANIFEST_KEY = "__neuroflame_artifacts__"
ARTIFACT_TAG = "neuroflame.artifact"
DEFAULT_ARTIFACT_CHUNK_BYTES = 1024 * 1024
MAX_PENDING_TRANSACTIONS = 1024

_TYPE_KEY = "__neuroflame_type__"
_VALUE_KEY = "value"
_TOKEN_KEY = "transfer_id"
_HASH_KEY = "sha256"
_SIZE_KEY = "size"
_NAME_KEY = "name"
_MEDIA_TYPE_KEY = "media_type"
_STAGE_KEY = "stage"
_DIRECTION_KEY = "direction"
_SOURCE_KEY = "source"
_STREAM_TRANSACTION_KEY = "neuroflame.artifact.stream_transaction"


class ArtifactTransferError(RuntimeError):
    """Report a local artifact transfer failure without artifact content."""

    def __init__(self, message: str, *, indeterminate: bool = False):
        """Record whether transport may still complete after this error."""
        super().__init__(message)
        self.indeterminate = indeterminate


@dataclass(frozen=True)
class _OutgoingArtifact:
    path: str
    name: str
    size: int
    sha256: str
    allowed_requesters: frozenset[str]
    stage: str
    direction: str


class _RetrievalWaiter(threading.Event):
    """Keep late 2.8 stream completion observable after caller timeout."""

    def __init__(self):
        super().__init__()
        self.result = None
        self.expired = False


class ArtifactTransfer(FLComponent):
    """Retrieve registered files by opaque ID and verify them before promotion."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_ARTIFACT_CHUNK_BYTES,
        chunk_timeout: float = 30.0,
    ):
        """Configure chunking while deferring all paths to the active run."""
        super().__init__()
        self.topic = "neuroflame_artifact_transfer_v1"
        self.stream_channel = self.__class__.__name__
        self.dest_dir = None
        self.chunk_size = chunk_size
        self.chunk_timeout = chunk_timeout
        self._root = ""
        self._artifact_root = ""
        self._output_artifact_root = ""
        self._incoming_dir = ""
        self._outgoing_dir = ""
        self._registry: Dict[str, _OutgoingArtifact] = {}
        self._lock = threading.RLock()
        self._shutting_down = False
        self._active_streams = 0
        self._transactions: Dict[str, _RetrievalWaiter] = {}
        self._cleanup_pending = False
        self._simulate_mode = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Register streaming for a run and securely clear staging at its end."""
        if event_type == EventType.START_RUN:
            self._initialize_run(fl_ctx)
            FileStreamer.register_stream_processing(
                fl_ctx=fl_ctx,
                channel=self.stream_channel,
                topic=self.topic,
                dest_dir=self.dest_dir,
                stream_done_cb=self._on_stream_done,
            )
            fl_ctx.get_engine().register_aux_message_handler(
                topic=self.topic,
                message_handle_func=self._handle_request,
            )
        elif event_type in (EventType.ABOUT_TO_END_RUN, EventType.END_RUN):
            with self._lock:
                self._shutting_down = True
            if event_type == EventType.END_RUN:
                self.cleanup(force=True)

    def _initialize_run(self, fl_ctx: FLContext) -> None:
        output_dir = os.path.realpath(resolve_output_directory(fl_ctx))
        self._output_artifact_root = os.path.join(output_dir, ".artifacts")
        temporary_parent = os.path.realpath(
            os.getenv("NEUROFLAME_ARTIFACT_TEMP_DIR") or tempfile.gettempdir()
        )
        try:
            self._artifact_root = tempfile.mkdtemp(
                prefix="neuroflame-artifacts-",
                dir=temporary_parent,
            )
            os.chmod(self._artifact_root, 0o700)
        except OSError as error:
            raise ArtifactTransferError(
                "Private artifact staging storage is unavailable"
            ) from error
        self._root = os.path.join(self._artifact_root, "transport")
        self._incoming_dir = os.path.join(self._root, "incoming")
        self._outgoing_dir = os.path.join(self._root, "outgoing")
        for directory in (self._root, self._incoming_dir, self._outgoing_dir):
            _ensure_private_directory(directory, self._artifact_root)
        self.dest_dir = self._incoming_dir
        self._shutting_down = False
        self._cleanup_pending = False
        self._simulate_mode = _is_nvflare_simulator(fl_ctx)

    def cleanup(self, *, force: bool = False) -> None:
        """Remove staging, clearing indeterminate tombstones at terminal shutdown."""
        with self._lock:
            if force:
                for waiter in self._transactions.values():
                    waiter.expired = True
                self._transactions.clear()
            if self._active_streams or (self._transactions and not force):
                self._cleanup_pending = True
                if force:
                    _remove_directory(self._incoming_dir)
                return
            self._registry.clear()
            self._cleanup_pending = False
            artifact_root = self._artifact_root
            output_artifact_root = self._output_artifact_root
        _remove_directory(artifact_root)
        _remove_directory(output_artifact_root)

    def _finish_activity(self) -> None:
        with self._lock:
            should_cleanup = (
                self._cleanup_pending
                and not self._active_streams
                and not self._transactions
            )
        if should_cleanup:
            self.cleanup(force=self._shutting_down)

    def discard_outgoing(self, tokens: Iterable[str]) -> None:
        """Remove snapshots registered while preparing a rejected payload."""
        paths = []
        with self._lock:
            for token in tokens:
                record = self._registry.pop(token, None)
                if record is not None:
                    paths.append(record.path)
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def register_outgoing(
        self,
        ref: ArtifactRef,
        *,
        source_root: str,
        allowed_requesters: Iterable[str],
        stage: str,
        direction: str,
        max_file_bytes: int,
        max_aggregate_bytes: int | None = None,
    ) -> Dict[str, Any]:
        """Stage a validated regular file and return metadata safe for Shareables."""
        with self._lock:
            if self._shutting_down:
                raise ArtifactTransferError("Artifact transfer is shutting down")
        validate_artifact_name(ref.name)
        requesters = frozenset(allowed_requesters)
        if not requesters or any(
            not isinstance(item, str) or not item for item in requesters
        ):
            raise ArtifactTransferError(
                "Artifact destination identities are unavailable"
            )

        source_path = _validate_source_path(ref.path, source_root)
        token = uuid.uuid4().hex
        staged_path = os.path.join(self._outgoing_dir, token)
        size, digest = _copy_regular_file(
            source_path,
            staged_path,
            max_file_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
        )
        record = _OutgoingArtifact(
            path=staged_path,
            name=ref.name,
            size=size,
            sha256=digest,
            allowed_requesters=requesters,
            stage=stage,
            direction=direction,
        )
        with self._lock:
            if self._shutting_down:
                _unlink_path(staged_path)
                raise ArtifactTransferError("Artifact transfer is shutting down")
            self._registry[token] = record
        return {
            _TOKEN_KEY: token,
            _NAME_KEY: ref.name,
            _MEDIA_TYPE_KEY: ref.media_type,
            _SIZE_KEY: size,
            _HASH_KEY: digest,
            _STAGE_KEY: stage,
            _DIRECTION_KEY: direction,
        }

    def validate_request(self, request: Shareable, fl_ctx: FLContext):
        """Authorize an opaque transfer request against the local registry."""
        token = request.get(_TOKEN_KEY)
        if not isinstance(token, str) or not _is_token(token):
            return ReturnCode.BAD_REQUEST_DATA, None
        with self._lock:
            record = self._registry.get(token)
        if record is None or self._shutting_down:
            return ReturnCode.BAD_REQUEST_DATA, None
        peer_ctx = fl_ctx.get_peer_context()
        peer = peer_ctx.get_identity_name() if peer_ctx else None
        if peer not in record.allowed_requesters:
            return ReturnCode.BAD_REQUEST_DATA, None
        expected = {
            _NAME_KEY: record.name,
            _SIZE_KEY: record.size,
            _HASH_KEY: record.sha256,
            _STAGE_KEY: record.stage,
            _DIRECTION_KEY: record.direction,
        }
        if any(request.get(key) != value for key, value in expected.items()):
            return ReturnCode.BAD_REQUEST_DATA, None
        return ReturnCode.OK, record

    def do_stream(
        self,
        target: str,
        request: Shareable,
        fl_ctx: FLContext,
        stream_ctx: StreamContext,
        validated_data: Any,
    ):
        """Stream the staged snapshot using NVFlare's blocking file API."""
        stream_target = (
            SiteType.SERVER if validated_data.direction == "site_to_central" else target
        )
        with self._lock:
            if self._shutting_down:
                raise ArtifactTransferError("Artifact transfer is shutting down")
            self._active_streams += 1
        try:
            return FileStreamer.stream_file(
                targets=[stream_target],
                stream_ctx=stream_ctx,
                channel=self.stream_channel,
                topic=self.topic,
                file_name=validated_data.path,
                fl_ctx=fl_ctx,
                chunk_size=self.chunk_size,
                chunk_timeout=self.chunk_timeout,
                optional=False,
                secure=not self._simulate_mode,
            )
        finally:
            with self._lock:
                self._active_streams -= 1
            self._finish_activity()

    def _handle_request(
        self, topic: str, request: Shareable, fl_ctx: FLContext
    ) -> Shareable:
        """Validate one retrieval request and stream it on an owned worker."""
        tx_id = request.get(_STREAM_TRANSACTION_KEY)
        if not isinstance(tx_id, str) or not _is_stream_transaction(tx_id):
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        peer_ctx = fl_ctx.get_peer_context()
        if not isinstance(peer_ctx, FLContext) or not peer_ctx.get_identity_name():
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        try:
            rc, validated_data = self.validate_request(request, fl_ctx)
        except Exception:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        if rc != ReturnCode.OK:
            return make_reply(rc)
        worker = threading.Thread(
            target=self._stream_for_request,
            args=(request, fl_ctx, validated_data),
            daemon=True,
        )
        worker.start()
        return make_reply(ReturnCode.OK)

    def _stream_for_request(
        self, request: Shareable, fl_ctx: FLContext, validated_data: Any
    ) -> None:
        """Run a blocking file stream without exposing request metadata in logs."""
        peer_ctx = fl_ctx.get_peer_context()
        target = peer_ctx.get_identity_name() if peer_ctx else ""
        stream_ctx = {_STREAM_TRANSACTION_KEY: request[_STREAM_TRANSACTION_KEY]}
        try:
            self.do_stream(target, request, fl_ctx, stream_ctx, validated_data)
        except Exception:
            self.log_error(fl_ctx, "Artifact stream failed")

    def retrieve(
        self, from_site: str, fl_ctx: FLContext, timeout: float, **obj_attrs
    ) -> tuple[str, Any, bool]:
        """Retrieve while retaining timed-out transactions for late cleanup.

        NVFlare 2.8 exposes no per-stream cancellation API. A timed-out waiter is
        therefore kept registered until a late stream completion can be observed
        and its temporary file removed. Callers must not retry indeterminate
        timeout or abort results.
        """
        engine = fl_ctx.get_engine()
        waiter = _RetrievalWaiter()
        tx_id = str(uuid.uuid4())
        with self._lock:
            if self._shutting_down:
                return ReturnCode.TASK_ABORTED, None, False
            if len(self._transactions) >= MAX_PENDING_TRANSACTIONS:
                return ReturnCode.EXECUTION_EXCEPTION, None, False
            self._transactions[tx_id] = waiter
        try:
            request = Shareable({_STREAM_TRANSACTION_KEY: tx_id})
            request.update(obj_attrs)
            replies = engine.send_aux_request(
                targets=[from_site],
                request=request,
                topic=self.topic,
                fl_ctx=fl_ctx,
                timeout=timeout,
            )
            reply = replies.get(from_site)
            if not isinstance(reply, Shareable):
                return self._indeterminate_result(
                    tx_id, waiter, ReturnCode.EXECUTION_EXCEPTION
                )
            rc = reply.get_return_code()
            if rc != ReturnCode.OK:
                self._finish_transaction(tx_id)
                return rc, None, False
            abort_signal = fl_ctx.get_run_abort_signal()
            deadline = time.monotonic() + timeout
            while not waiter.wait(0.1):
                if abort_signal and abort_signal.triggered:
                    if self._expire_waiter(waiter):
                        return ReturnCode.TASK_ABORTED, None, True
                    break
                if time.monotonic() >= deadline:
                    if self._expire_waiter(waiter):
                        return ReturnCode.TIMEOUT, None, True
                    break
            self._finish_transaction(tx_id)
            return waiter.result[0], waiter.result[1], False
        except Exception:
            return self._indeterminate_result(
                tx_id, waiter, ReturnCode.EXECUTION_EXCEPTION
            )

    def _indeterminate_result(
        self, tx_id: str, waiter: _RetrievalWaiter, return_code: str
    ) -> tuple[str, Any, bool]:
        """Retain a tombstone unless completion won the transport race."""
        if self._expire_waiter(waiter):
            return return_code, None, True
        result = waiter.result or (return_code, None)
        self._finish_transaction(tx_id)
        return result[0], result[1], False

    def _expire_waiter(self, waiter: _RetrievalWaiter) -> bool:
        """Atomically mark an unfinished retrieval as awaiting late cleanup."""
        with self._lock:
            if waiter.is_set():
                return False
            waiter.expired = True
            return True

    def _on_stream_done(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        """Deliver active results or securely discard completion after timeout."""
        tx_id = stream_ctx.get(_STREAM_TRANSACTION_KEY)
        location = FileStreamer.get_file_location(stream_ctx)
        try:
            result = (FileStreamer.get_rc(stream_ctx), location)
        except Exception:
            result = (ReturnCode.EXECUTION_EXCEPTION, location)
        with self._lock:
            waiter = self._transactions.get(tx_id)
        if not isinstance(waiter, _RetrievalWaiter):
            if result[1]:
                _remove_received_temp(result[1], self._incoming_dir)
            return
        with self._lock:
            expired = waiter.expired
            if not expired:
                waiter.result = result
                waiter.set()
        if expired:
            if result and len(result) == 2 and result[1]:
                _remove_received_temp(result[1], self._incoming_dir)
            self._finish_transaction(tx_id)

    def _finish_transaction(self, tx_id: str) -> None:
        with self._lock:
            self._transactions.pop(tx_id, None)
        self._finish_activity()

    def retrieve_artifact(
        self,
        *,
        from_site: str,
        manifest: Dict[str, Any],
        fl_ctx: FLContext,
        timeout: float,
        retries: int,
        max_file_bytes: int,
    ) -> str:
        """Retrieve, verify, and atomically promote one artifact."""
        _validate_manifest(manifest, max_file_bytes)
        token = manifest[_TOKEN_KEY]
        source_bucket = hashlib.sha256(from_site.encode("utf-8")).hexdigest()[:16]
        final_dir = os.path.join(self._incoming_dir, source_bucket)
        _ensure_private_directory(final_dir, self._incoming_dir)
        final_path = os.path.join(final_dir, f"{token}-{manifest[_NAME_KEY]}")
        if _verified_file(final_path, manifest[_SIZE_KEY], manifest[_HASH_KEY]):
            return final_path

        last_code = ReturnCode.ERROR
        last_integrity_error = None
        indeterminate = False
        for attempt in range(retries + 1):
            if self._shutting_down:
                raise ArtifactTransferError(
                    "Artifact transfer was cancelled during shutdown"
                )
            rc, temp_path, indeterminate = self.retrieve(
                from_site=from_site,
                fl_ctx=fl_ctx,
                timeout=timeout,
                **manifest,
            )
            last_code = rc
            if rc == ReturnCode.OK and temp_path:
                try:
                    promoted, verification_detail = _promote_verified_file_with_reason(
                        temp_path,
                        final_path,
                        manifest[_SIZE_KEY],
                        manifest[_HASH_KEY],
                    )
                    if promoted:
                        return final_path
                    last_integrity_error = (
                        f"Artifact integrity verification failed: {verification_detail}"
                    )
                finally:
                    _unlink_path(temp_path)
            if indeterminate or rc in (ReturnCode.TIMEOUT, ReturnCode.TASK_ABORTED):
                break
            if rc != ReturnCode.OK:
                break
            if attempt < retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))
        detail = last_integrity_error or f"return code {last_code}"
        raise ArtifactTransferError(
            f"Artifact transfer failed after {retries + 1} attempt(s): {detail}",
            indeterminate=indeterminate,
        )


def get_artifact_transfer(fl_ctx: FLContext) -> ArtifactTransfer:
    """Resolve the configured transfer component from the NVFlare engine."""
    component = fl_ctx.get_engine().get_component(ARTIFACT_COMPONENT_ID)
    if not isinstance(component, ArtifactTransfer):
        raise ArtifactTransferError(
            "Artifact transfer runtime component is unavailable"
        )
    return component


def prepare_outgoing_artifacts(
    value: Any,
    *,
    transfer: ArtifactTransfer,
    source_root: str,
    allowed_requesters: Iterable[str],
    stage: str,
    direction: str,
    max_file_bytes: int,
    max_total_bytes: int,
) -> tuple[Any, list[Dict[str, Any]]]:
    """Replace artifact references with safe tags and build a bounded manifest."""
    manifests = []
    staged_bytes = 0

    def visit(item):
        nonlocal staged_bytes
        if isinstance(item, ArtifactRef):
            remaining_bytes = max_total_bytes - staged_bytes
            manifest = transfer.register_outgoing(
                item,
                source_root=source_root,
                allowed_requesters=allowed_requesters,
                stage=stage,
                direction=direction,
                max_file_bytes=max_file_bytes,
                max_aggregate_bytes=remaining_bytes,
            )
            manifests.append(manifest)
            staged_bytes += manifest[_SIZE_KEY]
            return {_TYPE_KEY: ARTIFACT_TAG, _VALUE_KEY: dict(manifest)}
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, tuple):
            return tuple(visit(child) for child in item)
        if hasattr(item, "__dataclass_fields__") and not isinstance(item, type):
            from dataclasses import fields, replace

            return replace(
                item,
                **{
                    field.name: visit(getattr(item, field.name))
                    for field in fields(item)
                },
            )
        return item

    try:
        prepared = visit(value)
        return prepared, manifests
    except Exception:
        transfer.discard_outgoing(manifest[_TOKEN_KEY] for manifest in manifests)
        raise


def contains_artifacts(value: Any) -> bool:
    """Return whether a computation value contains an artifact reference."""
    if isinstance(value, ArtifactRef):
        return True
    if isinstance(value, dict):
        return any(contains_artifacts(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_artifacts(item) for item in value)
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        from dataclasses import fields

        return any(
            contains_artifacts(getattr(value, field.name)) for field in fields(value)
        )
    return False


def materialize_incoming_artifacts(
    value: Any,
    manifests: Any,
    *,
    transfer: ArtifactTransfer,
    from_site: str,
    fl_ctx: FLContext,
    expected_stage: str,
    expected_direction: str,
    timeout: float,
    retries: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> Any:
    """Retrieve every declared artifact and replace tags with local references."""
    if not manifests:
        return value
    by_token, total_size = validate_artifact_manifests(
        manifests,
        expected_stage=expected_stage,
        expected_direction=expected_direction,
        max_file_bytes=max_file_bytes,
    )
    if total_size > max_total_bytes:
        raise ArtifactTransferError(
            "Aggregate artifact transfer size exceeds the configured limit"
        )

    referenced = set()

    def validate_tags(item):
        if isinstance(item, dict) and item.get(_TYPE_KEY) == ARTIFACT_TAG:
            metadata = item.get(_VALUE_KEY)
            if not isinstance(metadata, dict):
                raise ArtifactTransferError("Artifact payload tag is invalid")
            token = metadata.get(_TOKEN_KEY)
            manifest = by_token.get(token)
            if manifest is None or metadata != manifest:
                raise ArtifactTransferError(
                    "Artifact payload does not match its manifest"
                )
            referenced.add(token)
            return
        if isinstance(item, dict):
            for child in item.values():
                validate_tags(child)
        elif isinstance(item, list):
            for child in item:
                validate_tags(child)

    validate_tags(value)
    if referenced != set(by_token):
        raise ArtifactTransferError(
            "Artifact manifest contains an unreferenced transfer"
        )

    materialized = {}
    try:
        for token, manifest in by_token.items():
            materialized[token] = transfer.retrieve_artifact(
                from_site=from_site,
                manifest=manifest,
                fl_ctx=fl_ctx,
                timeout=timeout,
                retries=retries,
                max_file_bytes=max_file_bytes,
            )
    except Exception:
        for path in materialized.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise

    seen = set()

    def visit(item):
        if isinstance(item, dict) and item.get(_TYPE_KEY) == ARTIFACT_TAG:
            metadata = item.get(_VALUE_KEY)
            if not isinstance(metadata, dict):
                raise ArtifactTransferError("Artifact payload tag is invalid")
            token = metadata.get(_TOKEN_KEY)
            manifest = by_token.get(token)
            if manifest is None or metadata != manifest:
                raise ArtifactTransferError(
                    "Artifact payload does not match its manifest"
                )
            seen.add(token)
            return ArtifactRef(
                name=manifest[_NAME_KEY],
                path=materialized[token],
                media_type=manifest.get(_MEDIA_TYPE_KEY),
            )
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    result = visit(value)
    if seen != set(by_token):
        raise ArtifactTransferError(
            "Artifact manifest contains an unreferenced transfer"
        )
    return result


def validate_artifact_manifests(
    manifests: Any,
    *,
    expected_stage: str,
    expected_direction: str,
    max_file_bytes: int,
) -> tuple[Dict[str, Dict[str, Any]], int]:
    """Validate metadata without initiating transfers and return its byte total."""
    if not isinstance(manifests, list):
        raise ArtifactTransferError("Artifact manifest is invalid")
    by_token = {}
    total_size = 0
    for manifest in manifests:
        _validate_manifest(manifest, max_file_bytes)
        if (
            manifest[_STAGE_KEY] != expected_stage
            or manifest[_DIRECTION_KEY] != expected_direction
        ):
            raise ArtifactTransferError("Artifact stage or direction is invalid")
        token = manifest[_TOKEN_KEY]
        if token in by_token:
            raise ArtifactTransferError(
                "Artifact manifest contains a duplicate transfer"
            )
        by_token[token] = manifest
        total_size += manifest[_SIZE_KEY]
    return by_token, total_size


def _validate_source_path(path: str, source_root: str) -> str:
    lexical_root = Path(os.path.abspath(source_root))
    source = Path(os.path.abspath(path))
    try:
        source.relative_to(lexical_root)
    except ValueError as error:
        raise ArtifactTransferError(
            "Artifact source is outside the runtime staging root"
        ) from error
    current = source
    while current != lexical_root:
        if current.is_symlink():
            raise ArtifactTransferError("Artifact source path cannot contain symlinks")
        current = current.parent
    root = Path(source_root).resolve(strict=True)
    resolved = source.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ArtifactTransferError(
            "Artifact source is outside the runtime staging root"
        )
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ArtifactTransferError("Artifact source must be a regular file")
    return str(resolved)


def _copy_regular_file(
    source: str,
    destination: str,
    max_bytes: int,
    *,
    max_aggregate_bytes: int | None = None,
) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ArtifactTransferError("Artifact source must be a regular file")
        if source_stat.st_nlink != 1:
            raise ArtifactTransferError("Artifact source cannot be a hard-linked file")
        if source_stat.st_size > max_bytes:
            raise ArtifactTransferError(
                "Artifact exceeds the configured file size limit"
            )
        if (
            max_aggregate_bytes is not None
            and source_stat.st_size > max_aggregate_bytes
        ):
            raise ArtifactTransferError(
                "Aggregate artifact transfer size exceeds the configured limit"
            )
        destination_fd = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            while True:
                chunk = os.read(source_fd, DEFAULT_ARTIFACT_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactTransferError(
                        "Artifact exceeds the configured file size limit"
                    )
                if max_aggregate_bytes is not None and size > max_aggregate_bytes:
                    raise ArtifactTransferError(
                        "Aggregate artifact transfer size exceeds the configured limit"
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        final_stat = os.fstat(source_fd)
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size) != (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
        ):
            raise ArtifactTransferError("Artifact source changed while it was staged")
    except Exception:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_fd)
    return size, digest.hexdigest()


def _verified_file(path: str, expected_size: int, expected_hash: str) -> bool:
    verified = _open_verified_file(path, expected_size, expected_hash)
    if verified is None:
        return False
    descriptor, _ = verified
    os.close(descriptor)
    return True


def _open_verified_file(
    path: str, expected_size: int, expected_hash: str
) -> tuple[int, os.stat_result] | None:
    """Hash and validate one non-symlink regular file through one descriptor."""
    verified, _ = _open_verified_file_with_reason(path, expected_size, expected_hash)
    return verified


def _open_verified_file_with_reason(
    path: str, expected_size: int, expected_hash: str
) -> tuple[tuple[int, os.stat_result] | None, str | None]:
    """Verify a file and return a content-free diagnostic on failure."""
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(descriptor)
            return None, "received path is not a regular file"
        if file_stat.st_nlink != 1:
            os.close(descriptor)
            return (
                None,
                f"received file has link count {file_stat.st_nlink}, expected 1",
            )
        if file_stat.st_size != expected_size:
            os.close(descriptor)
            return (
                None,
                f"received size {file_stat.st_size} does not match expected "
                f"size {expected_size}",
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, DEFAULT_ARTIFACT_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        final_stat = os.fstat(descriptor)
        identity = (file_stat.st_dev, file_stat.st_ino, file_stat.st_size)
        if identity != (final_stat.st_dev, final_stat.st_ino, final_stat.st_size):
            os.close(descriptor)
            return None, "received file identity changed during verification"
        if digest.hexdigest() != expected_hash:
            os.close(descriptor)
            return None, "received SHA-256 does not match the manifest"
        return (descriptor, final_stat), None
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        return None, _safe_os_error("opening or reading the received file", error)


def _promote_verified_file(
    temporary_path: str,
    final_path: str,
    expected_size: int,
    expected_hash: str,
) -> bool:
    """Promote exactly the inode that was hashed, detecting pathname swaps."""
    promoted, _ = _promote_verified_file_with_reason(
        temporary_path,
        final_path,
        expected_size,
        expected_hash,
    )
    return promoted


def _promote_verified_file_with_reason(
    temporary_path: str,
    final_path: str,
    expected_size: int,
    expected_hash: str,
) -> tuple[bool, str | None]:
    """Promote a verified inode and return a content-free failure diagnostic."""
    verified, failure = _open_verified_file_with_reason(
        temporary_path, expected_size, expected_hash
    )
    if verified is None:
        return False, failure
    descriptor, verified_stat = verified
    identity = (verified_stat.st_dev, verified_stat.st_ino)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as error:
            return False, _safe_os_error("setting private file permissions", error)
        try:
            current = os.lstat(temporary_path)
        except OSError as error:
            return False, _safe_os_error("checking the received path", error)
        if (
            not stat.S_ISREG(current.st_mode)
            or (
                current.st_dev,
                current.st_ino,
            )
            != identity
        ):
            return False, "received path identity changed before promotion"
        try:
            os.replace(temporary_path, final_path)
        except OSError as error:
            return False, _safe_os_error(
                "atomically promoting the received file", error
            )
        try:
            promoted = os.lstat(final_path)
        except OSError as error:
            return False, _safe_os_error("checking the promoted file", error)
        if (
            not stat.S_ISREG(promoted.st_mode)
            or (
                promoted.st_dev,
                promoted.st_ino,
            )
            != identity
        ):
            _unlink_path(final_path)
            return False, "promoted file identity does not match the verified file"
        return True, None
    finally:
        os.close(descriptor)


def _safe_os_error(operation: str, error: OSError) -> str:
    """Describe a filesystem failure without including a sensitive pathname."""
    errno_value = error.errno if error.errno is not None else "unknown"
    description = error.strerror or type(error).__name__
    return f"{operation} failed (errno {errno_value}: {description})"


def _unlink_path(path: str) -> None:
    """Unlink a pathname itself without following a final symlink."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _remove_received_temp(path: str, incoming_root: str) -> None:
    """Remove only a late temporary path owned by the incoming staging root."""
    candidate = Path(os.path.abspath(path))
    root = Path(os.path.abspath(incoming_root))
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    _unlink_path(str(candidate))


def _remove_directory(path: str) -> None:
    """Remove one runtime-owned staging tree if it still exists."""
    if not path:
        return
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(path)


def _is_nvflare_simulator(fl_ctx: FLContext) -> bool:
    """Recognize only NVFlare's explicit simulator context or engine classes."""
    get_prop = getattr(fl_ctx, "get_prop", None)
    if (
        callable(get_prop)
        and get_prop(FLContextKey.SIMULATE_MODE, default=False) is True
    ):
        return True
    get_engine = getattr(fl_ctx, "get_engine", None)
    engine = get_engine() if callable(get_engine) else None
    if engine is None:
        return False
    engine_type = type(engine)
    qualified_name = f"{engine_type.__module__}.{engine_type.__qualname__}".lower()
    return qualified_name.startswith("nvflare.") and "simulator" in qualified_name


def _validate_manifest(manifest: Any, max_file_bytes: int) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactTransferError("Artifact manifest entry is invalid")
    required = {
        _TOKEN_KEY,
        _NAME_KEY,
        _MEDIA_TYPE_KEY,
        _SIZE_KEY,
        _HASH_KEY,
        _STAGE_KEY,
        _DIRECTION_KEY,
    }
    if set(manifest) != required:
        raise ArtifactTransferError("Artifact manifest entry is incomplete")
    if not _is_token(manifest[_TOKEN_KEY]):
        raise ArtifactTransferError("Artifact transfer identifier is invalid")
    validate_artifact_name(manifest[_NAME_KEY])
    media_type = manifest[_MEDIA_TYPE_KEY]
    if media_type is not None and not isinstance(media_type, str):
        raise ArtifactTransferError("Artifact media type is invalid")
    size = manifest[_SIZE_KEY]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size > max_file_bytes
    ):
        raise ArtifactTransferError(
            "Artifact size is invalid or exceeds the configured limit"
        )
    digest = manifest[_HASH_KEY]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ArtifactTransferError("Artifact integrity hash is invalid")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ArtifactTransferError("Artifact integrity hash is invalid") from error
    if not isinstance(manifest[_STAGE_KEY], str) or not manifest[_STAGE_KEY]:
        raise ArtifactTransferError("Artifact stage is invalid")
    if manifest[_DIRECTION_KEY] not in {"site_to_central", "central_to_site"}:
        raise ArtifactTransferError("Artifact direction is invalid")


def _is_token(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_stream_transaction(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _ensure_private_directory(path: str, allowed_root: str) -> None:
    root = Path(allowed_root).resolve(strict=True)
    candidate = Path(path).absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ArtifactTransferError(
            "Artifact staging directory is outside its runtime root"
        ) from error
    current = candidate
    missing = []
    while current != root:
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ArtifactTransferError(
                    "Artifact staging path is not a private directory"
                )
        else:
            missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    os.chmod(candidate, 0o700)
