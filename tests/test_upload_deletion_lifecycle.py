from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "apps/api/src/upload_service.rs"
BINARY = ROOT / "apps/api/src/simple_core.rs"
QUOTA = ROOT / "apps/api/src/quota.rs"
ACCOUNT = ROOT / "apps/api/src/account_worker.rs"
ACCOUNT_SERVICE = ROOT / "apps/api/src/account_service.rs"
MIGRATION = ROOT / "apps/api/migrations/0041_upload_deletion_fencing.sql"
BACKUP_ERASURE_MIGRATION = (
    ROOT / "apps/api/migrations/0046_verified_backup_erasure.sql"
)
EXPORT_DELETE_MIGRATION = (
    ROOT / "apps/api/migrations/0047_resumable_account_export_deletion.sql"
)
COMPLETED_UPLOAD_LEASE_MIGRATION = (
    ROOT / "apps/api/migrations/0048_completed_upload_finalization_lease.sql"
)
LIVE_ALPHA = ROOT / "tests/live_alpha_safety.py"


class UploadDeletionLifecycleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upload = UPLOAD.read_text(encoding="utf-8")
        cls.binary = BINARY.read_text(encoding="utf-8")
        cls.quota = QUOTA.read_text(encoding="utf-8")
        cls.account = ACCOUNT.read_text(encoding="utf-8")
        cls.account_service = ACCOUNT_SERVICE.read_text(encoding="utf-8")
        cls.migration = MIGRATION.read_text(encoding="utf-8")
        cls.backup_erasure_migration = BACKUP_ERASURE_MIGRATION.read_text(
            encoding="utf-8"
        )
        cls.export_delete_migration = EXPORT_DELETE_MIGRATION.read_text(
            encoding="utf-8"
        )
        cls.completed_upload_lease_migration = (
            COMPLETED_UPLOAD_LEASE_MIGRATION.read_text(encoding="utf-8")
        )
        cls.live_alpha = LIVE_ALPHA.read_text(encoding="utf-8")

    def test_deletion_transition_installs_an_immutable_storage_fence(self) -> None:
        self.assertIn("CREATE TABLE brunn.account_deletion_fences", self.migration)
        self.assertIn("account_deletion_fences_immutable", self.migration)
        transition = self.migration.index("OLD.account_status = 'active'")
        storage_lock = self.migration.index(
            "hashtextextended('storage:' || NEW.id::text, 0)", transition
        )
        fence_insert = self.migration.index(
            "INSERT INTO brunn.account_deletion_fences", storage_lock
        )
        self.assertLess(transition, storage_lock)
        self.assertLess(storage_lock, fence_insert)
        self.assertIn("a deletion-fenced account cannot be reactivated", self.migration)

    def test_binary_upload_hashes_bytes_before_atomic_publication(self) -> None:
        upload = self.binary[
            self.binary.index("pub async fn upload_binary") :
            self.binary.index("pub async fn upload_binary_stream")
        ]
        path = upload.index("validate_public_path(&path)?")
        expected_hash = upload.index('"expected_content_hash is required"', path)
        object_write = upload.index(".put_user_blob(", expected_hash)
        stored_hash = upload.index("validate_sha256(&stored.sha256)?", object_write)
        mismatch = upload.index('"content_hash_mismatch"', stored_hash)
        publish = upload.index("commit_binary_with_companion(", mismatch)
        self.assertLess(path, expected_hash)
        self.assertLess(expected_hash, object_write)
        self.assertLess(object_write, stored_hash)
        self.assertLess(stored_hash, mismatch)
        self.assertLess(mismatch, publish)

    def test_streamed_upload_is_bounded_and_always_unlinks_its_buffer(self) -> None:
        stream = self.binary[
            self.binary.index("pub async fn upload_binary_stream") :
            self.binary.index("pub async fn fetch_binary")
        ]
        self.assertIn("auth.require(Capability::Stage)?", stream)
        self.assertIn("validate_sha256(&query.expected_content_hash)?", stream)
        self.assertIn(".create_new(true)", stream)
        self.assertIn("checked_add", stream)
        self.assertIn("MAX_STREAMED_BINARY_BYTES", stream)
        self.assertIn("put_user_file_blob", stream)
        transfer = stream.index("let transfer = async")
        cleanup = stream.index("tokio::fs::remove_file(&temporary_path)", transfer)
        unwrap = stream.index("let stored = transfer?", cleanup)
        hash_check = stream.index('stored.sha256.trim_start_matches("sha256:")', unwrap)
        publish = stream.index("commit_binary_with_companion(", hash_check)
        self.assertLess(transfer, cleanup)
        self.assertLess(cleanup, unwrap)
        self.assertLess(unwrap, hash_check)
        self.assertLess(hash_check, publish)

    def test_binary_commit_locks_paths_and_fences_versions(self) -> None:
        commit = self.binary[self.binary.index("async fn commit_binary_with_companion") :]
        exact_object = commit.index("object_version_id.ok_or_else")
        transaction = commit.index("let mut tx = state.begin_write(auth).await?", exact_object)
        lock = commit.index("require_local_publish_lock", transaction)
        row_lock = commit.index("FOR UPDATE OF entry", lock)
        version_check = commit.index("entry_version_conflict", row_lock)
        version_insert = commit.index("INSERT INTO brunn.entry_versions", version_check)
        head_update = commit.index("UPDATE brunn.entries", version_insert)
        generation = commit.index("INSERT INTO brunn.workspace_changes", head_update)
        durable = commit.index("tx.commit().await?", generation)
        self.assertLess(exact_object, transaction)
        self.assertLess(transaction, lock)
        self.assertLess(lock, row_lock)
        self.assertLess(row_lock, version_check)
        self.assertLess(version_check, version_insert)
        self.assertLess(version_insert, head_update)
        self.assertLess(head_update, generation)
        self.assertLess(generation, durable)

    def test_account_multipart_abort_verifies_the_prefix_is_empty(self) -> None:
        abort = self.upload.index("pub async fn abort_multipart_prefix")
        validate = self.upload.index("validate_user_prefix(prefix)?", abort)
        first_list = self.upload.index("list_multipart_uploads(state, Some(prefix))", validate)
        abort_object = self.upload.index(".abort_multipart_upload(", first_list)
        second_list = self.upload.index(
            "list_multipart_uploads(state, Some(prefix))", first_list + 1
        )
        remaining = self.upload.index("if !remaining.is_empty()", second_list)
        self.assertLess(validate, first_list)
        self.assertLess(first_list, abort_object)
        self.assertLess(abort_object, second_list)
        self.assertLess(second_list, remaining)
        self.assertIn(".list_multipart_uploads()", self.upload)
        self.assertIn("next_key_marker", self.upload)
        self.assertIn("next_upload_id_marker", self.upload)

    def test_historical_completed_upload_rows_retain_exact_lease_constraints(self) -> None:
        self.assertIn(
            "OR status IN ('verifying','completed')",
            self.completed_upload_lease_migration,
        )
        self.assertIn(
            "(completion_token IS NULL) = (completion_lease_expires_at IS NULL)",
            self.completed_upload_lease_migration,
        )

    def test_account_deletion_holds_lock_through_two_storage_sweeps(self) -> None:
        advance = self.account.index("async fn advance_account_deletion")
        lock = self.account.index("pg_advisory_xact_lock", advance)
        abort_first = self.account.index("abort_multipart_prefix", lock)
        purge_first = self.account.index("purge_prefix", abort_first)
        abort_second = self.account.index("abort_multipart_prefix", purge_first)
        purge_second = self.account.index("purge_prefix", abort_second)
        redact = self.account.index("redact_account_for_retention", purge_second)
        commit = self.account.index("tx.commit().await?", redact)
        self.assertLess(lock, abort_first)
        self.assertLess(abort_first, purge_first)
        self.assertLess(purge_first, abort_second)
        self.assertLess(abort_second, purge_second)
        self.assertLess(purge_second, redact)
        self.assertLess(redact, commit)

    def test_account_export_is_streamed_and_published_under_the_fence(self) -> None:
        publish = self.account.index("async fn publish_account_export")
        lock = self.account.index("assert_storage_write_allowed", publish)
        create = self.account.index(".create_multipart_upload(", lock)
        parts = self.account.index("upload_export_parts(", create)
        complete = self.account.index(".complete_multipart_upload(", parts)
        ready = self.account.index("SET status='ready'", complete)
        commit = self.account.index("tx.commit().await", ready)
        self.assertLess(lock, create)
        self.assertLess(create, parts)
        self.assertLess(parts, complete)
        self.assertLess(complete, ready)
        self.assertLess(ready, commit)
        self.assertIn("ACCOUNT_EXPORT_PART_SIZE_BYTES", self.account)
        self.assertNotIn(".put_file(", self.account[publish:])

    def test_account_export_uses_a_separate_bounded_temporary_allowance(self) -> None:
        publish = self.account.index("async fn publish_account_export")
        allowance = self.account.index(
            "ensure_temporary_export_capacity", publish
        )
        multipart = self.account.index(".create_multipart_upload(", allowance)
        self.assertLess(allowance, multipart)

        durable_quota = self.quota[
            self.quota.index("pub async fn ensure_storage_capacity_for_objects")
            : self.quota.index("pub async fn ensure_temporary_export_capacity")
        ]
        temporary_start = self.quota.index(
            "pub async fn ensure_temporary_export_capacity"
        )
        temporary_quota = self.quota[
            temporary_start : self.quota.index(
                "impl UsageReservation", temporary_start
            )
        ]
        self.assertNotIn("brunn.account_exports", durable_quota)
        self.assertIn("temporary_export_limit(storage_limit)", temporary_quota)
        self.assertIn("TEMPORARY_EXPORT_MAX_OVERHEAD_BYTES", self.quota)

    def test_account_export_download_stream_enforces_size_and_sha256(self) -> None:
        download = self.account_service.index("pub async fn download_export")
        exact_version = self.account_service.index(
            ".get_stream_version(&object_key, Some(&object_version_id))", download
        )
        head_size = self.account_service.index(
            "object.content_length !=", exact_version
        )
        verifier = self.account_service.index(
            "AccountExportIntegrityReader::new(", head_size
        )
        body = self.account_service.index("Body::from_stream(stream)", verifier)
        self.assertLess(exact_version, head_size)
        self.assertLess(head_size, verifier)
        self.assertLess(verifier, body)
        self.assertIn(
            "account export stream failed size or SHA-256 verification",
            self.account_service,
        )

    def test_export_delete_is_durable_before_object_purge(self) -> None:
        delete = self.account_service.index("pub async fn delete_export")
        begin = self.account_service.index(
            "brunn_auth.begin_account_export_delete", delete
        )
        durable = self.account_service.index("tx.commit().await?", begin)
        purge = self.account_service.index(".purge_all_versions(", durable)
        finish = self.account_service.index(
            "brunn_auth.finish_account_export_delete", purge
        )
        completed = self.account_service.index("tx.commit().await?", finish)
        self.assertLess(begin, durable)
        self.assertLess(durable, purge)
        self.assertLess(purge, finish)
        self.assertLess(finish, completed)
        self.assertIn("SECURITY DEFINER", self.export_delete_migration)
        self.assertIn("FOR UPDATE", self.export_delete_migration)
        self.assertIn(
            "brunn_auth.require_credential_control",
            self.export_delete_migration,
        )
        self.assertIn("SET status='deleting'", self.export_delete_migration)
        self.assertIn("TO app_rw", self.export_delete_migration)
        self.assertIn(
            "DROP FUNCTION brunn_auth.lock_account_export_for_delete",
            self.export_delete_migration,
        )

    def test_account_deletion_requires_verified_backup_erasure(self) -> None:
        self.assertIn(
            "account_deletion_completion_requires_backup_proof",
            self.backup_erasure_migration,
        )
        self.assertIn("canonical_purged_at", self.account)
        self.assertIn("backup_erasure_verified", self.account)
        self.assertIn(
            "account deletion cannot complete without verified backup erasure",
            self.account,
        )
        self.assertIn("record_backup_erasure_proof", self.live_alpha)
        self.assertIn('str(env_file.resolve())', self.live_alpha)

    def test_unready_backup_retention_does_not_starve_actionable_deletions(self) -> None:
        claim = self.account.index("async fn claim_account_deletion")
        prepare = self.account.index("async fn prepare_account_deletion", claim)
        claim_body = self.account[claim:prepare]
        self.assertIn("status IN ('queued','running')", claim_body)
        self.assertIn(
            "status='awaiting_backup_expiry'\n"
            "              AND backup_expiry_due_at <= clock_timestamp()",
            claim_body,
        )
        self.assertIn(
            "CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END",
            claim_body,
        )

    def test_account_export_includes_legacy_and_simple_binary_versions(self) -> None:
        query = self.account.index("const ACCOUNT_EXPORT_OBJECT_REFERENCES_SQL")
        build = self.account.index("async fn build_export_inner")
        query_use = self.account.index(
            "sqlx::query(ACCOUNT_EXPORT_OBJECT_REFERENCES_SQL)", build
        )
        asset_versions = self.account.index(
            "FROM brunn.asset_versions", query
        )
        union = self.account.index("UNION ALL", asset_versions)
        entry_versions = self.account.index(
            "FROM brunn.entry_versions", union
        )
        exact_dedupe = self.account.index(
            "GROUP BY bucket,object_key,object_version_id", entry_versions
        )
        download = self.account.index(
            ".download_version_to_path(", exact_dedupe
        )
        hash_check = self.account.index(
            "export object integrity check failed", download
        )
        self.assertLess(query, asset_versions)
        self.assertLess(asset_versions, union)
        self.assertLess(union, entry_versions)
        self.assertLess(entry_versions, exact_dedupe)
        self.assertLess(exact_dedupe, download)
        self.assertLess(download, hash_check)
        self.assertLess(query, build)
        self.assertLess(build, query_use)
        self.assertIn("object_key IS NOT NULL", self.account[entry_versions:exact_dedupe])
        self.assertIn("object_version_id IS NOT NULL", self.account[entry_versions:exact_dedupe])
        self.assertIn("'entry_version'::text", self.account[union:exact_dedupe])
        self.assertIn("count(DISTINCT content_hash)", self.account[query:exact_dedupe])
        self.assertIn("count(DISTINCT size_bytes)", self.account[query:exact_dedupe])
        self.assertIn(
            "application/x-brunn-deleted",
            self.account[query:union],
        )
        self.assertIn(
            "redacted_by_deletion_job",
            self.account[query:union],
        )

    def test_live_alpha_exports_and_purges_simple_workspace_binary_bytes(self) -> None:
        self.assertIn('"/v1/workspace/binaries/content?', self.live_alpha)
        self.assertIn("expected_content_hash", self.live_alpha)
        self.assertIn('"kind": "entry_version"', self.live_alpha)
        self.assertIn("object_file.read() == binary_bytes", self.live_alpha)
        self.assertIn('awaiting["result"]["object_versions_deleted"] >= 1', self.live_alpha)
        self.assertIn('"storage_reconciliation_passes"', self.live_alpha)

    def test_live_alpha_keeps_schema_derived_purge_and_backup_gates(self) -> None:
        self.assertIn("assert_nonreceipt_user_rows_are_purged", self.live_alpha)
        self.assertIn("record_backup_erasure_proof", self.live_alpha)
        self.assertIn("only_status_credential_active", self.live_alpha)
        self.assertIn("backup_status", self.live_alpha)
        self.assertIn("retained_until_deadline", self.live_alpha)


if __name__ == "__main__":
    unittest.main()
