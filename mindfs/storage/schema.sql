-- MindFS Authoritative SQLite Schema

CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime REAL NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT,
    mime_type TEXT NOT NULL,
    category TEXT NOT NULL,
    processor TEXT,
    processor_version TEXT,
    status TEXT NOT NULL,
    status_reason TEXT,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_offset TEXT,
    text TEXT NOT NULL,
    summary TEXT,
    metadata TEXT,
    entities TEXT,
    created_at TEXT NOT NULL,
    processor TEXT NOT NULL,
    processor_version TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    vector_id INTEGER,
    source_path TEXT NOT NULL,
    source_offset TEXT,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    metadata TEXT,
    FOREIGN KEY(artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS index_runs (
    run_id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT,
    files_scanned INTEGER DEFAULT 0,
    files_indexed INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    files_unsupported INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    artifacts_created INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    peak_rss_mb REAL DEFAULT 0.0,
    duration_seconds REAL DEFAULT 0.0,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    processor TEXT,
    error_message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnostics (
    diag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    peak_rss_mb REAL NOT NULL,
    current_rss_mb REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    files_processed INTEGER DEFAULT 0,
    bytes_processed INTEGER DEFAULT 0,
    chunks_processed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexed_folders (
    folder_path TEXT PRIMARY KEY,
    folder_name TEXT NOT NULL,
    added_at TEXT NOT NULL,
    last_indexed_at TEXT,
    files_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id TEXT PRIMARY KEY,
    plan_id TEXT,
    action_type TEXT NOT NULL,
    source_path TEXT,
    destination_path TEXT,
    backup_path TEXT,
    details TEXT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    undone INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    thoughts TEXT,
    tool_calls TEXT,
    explored_files TEXT,
    subagents TEXT,
    plan_id TEXT,
    plan_data TEXT,
    status TEXT,
    can_undo INTEGER DEFAULT 0,
    undo_log_ids TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_file_id ON artifacts(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_artifact_id ON chunks(artifact_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_audit_plan_id ON audit_logs(plan_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS recent_workspaces (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);


