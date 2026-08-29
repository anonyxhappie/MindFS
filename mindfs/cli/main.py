"""Command-line interface for MindFS."""

import json
from pathlib import Path
import sys
import click
import yaml

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools
from mindfs.config.settings import MindFSConfig, load_config
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.indexer import Indexer
from mindfs.indexing.vector_store import VectorStore
from mindfs.processors import create_default_registry
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore


def build_system_context(config_path: str = None, workspace_root: str = None):
    """Factory helper to wire up all MindFS engine components."""
    config = load_config(config_path=config_path, workspace_root=workspace_root)
    sandbox = FilesystemSandbox(config.resolved_workspace_root)
    resources = ResourceManager(config)
    store = SQLiteStore(config.resolved_db_path)
    registry = create_default_registry(config)
    vector_store = VectorStore(
        embedding_dim=config.embedding.embedding_dim,
        index_path=config.resolved_faiss_path,
    )
    embedding_pipeline = EmbeddingPipeline(config)
    indexer = Indexer(
        config=config,
        sandbox=sandbox,
        store=store,
        registry=registry,
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        resource_manager=resources,
    )
    search_engine = SearchEngine(
        config=config,
        sandbox=sandbox,
        store=store,
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        resource_manager=resources,
    )
    tools = FilesystemTools(
        config=config,
        sandbox=sandbox,
        store=store,
        indexer=indexer,
        search_engine=search_engine,
        registry=registry,
    )
    llm_engine = LLMEngine(config)
    agent = MindFSAgent(config=config, tools=tools, llm_engine=llm_engine)

    return {
        "config": config,
        "sandbox": sandbox,
        "resources": resources,
        "store": store,
        "registry": registry,
        "vector_store": vector_store,
        "indexer": indexer,
        "search_engine": search_engine,
        "tools": tools,
        "llm_engine": llm_engine,
        "agent": agent,
    }


@click.group()
@click.option("--config", "-c", "config_path", help="Path to config.yaml")
@click.option("--workspace", "-w", "workspace_root", help="Workspace root directory")
@click.pass_context
def cli(ctx, config_path, workspace_root):
    """MindFS: Local, Privacy-First Filesystem Intelligence Engine."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["workspace_root"] = workspace_root


@cli.command("init")
@click.option("--workspace", "-w", default=".", help="Target workspace root")
@click.pass_context
def init_cmd(ctx, workspace):
    """Initializes MindFS inside the specified workspace."""
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    mindfs_dir = ws / ".mindfs"
    mindfs_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = ws / "config.yaml"
    if not cfg_file.exists():
        default_cfg = {
            "workspace_root": str(ws),
            "index": {
                "db_path": ".mindfs/metadata.db",
                "faiss_path": ".mindfs/index.faiss",
                "max_file_size_mb": 5.0,
            },
            "resources": {
                "max_rss_mb": 2048.0,
            },
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(default_cfg, f)

    click.echo(f"Initialized MindFS workspace at: {ws}")


@cli.command("status")
@click.pass_context
def status_cmd(ctx):
    """Shows index statistics, vector count, and database size."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    tools = ctx_objs["tools"]
    res = tools.get_index_status()
    if res.success and res.data:
        st = res.data
        click.echo("================ MindFS Index Status ================")
        click.echo(f"Workspace Root   : {ctx_objs['config'].resolved_workspace_root}")
        click.echo(f"Files Indexed    : {st.get('files_indexed')} / {st.get('files_total')}")
        click.echo(f"Completed        : {st.get('files_completed')}")
        click.echo(f"Skipped          : {st.get('files_skipped')}")
        click.echo(f"Failed           : {st.get('files_failed')}")
        click.echo(f"Artifacts Count  : {st.get('artifacts_count')}")
        click.echo(f"Chunks Count     : {st.get('chunks_count')}")
        click.echo(f"Vectors (FAISS)  : {st.get('vectors_count')}")
        click.echo(f"Database Size    : {st.get('db_size_mb')} MB")
        click.echo("====================================================")
    else:
        click.echo(f"Error fetching status: {res.error}")


@cli.command("list")
@click.argument("path", required=False, default="")
@click.pass_context
def list_cmd(ctx, path):
    """Lists files and directories safely inside the workspace."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    tools = ctx_objs["tools"]
    res = tools.list_directory(path)
    if res.success and res.data:
        click.echo(f"Contents of '{path or '.'}':")
        for item in res.data:
            kind = "DIR " if item["type"] == "directory" else "FILE"
            click.echo(f"  [{kind}] {item['name']:<30} {item['size_bytes']:>10} bytes")
    else:
        click.echo(f"Error: {res.error}")


@cli.command("inspect")
@click.argument("path", required=True)
@click.pass_context
def inspect_cmd(ctx, path):
    """Performs lightweight technical inspection of a file."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    tools = ctx_objs["tools"]
    res = tools.inspect_file(path)
    if res.success and res.data:
        click.echo(json.dumps(res.data, indent=2))
    else:
        click.echo(f"Error inspecting file: {res.error}")


@cli.command("index")
@click.argument("path", required=False, default="")
@click.option("--recursive/--no-recursive", default=True, help="Scan directories recursively")
@click.pass_context
def index_cmd(ctx, path, recursive):
    """Indexes a file or directory incrementally."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    tools = ctx_objs["tools"]
    click.echo(f"Indexing target '{path or '.'}' (recursive={recursive})...")
    res = tools.index_path(path=path, recursive=recursive)
    if res.success and res.data:
        d = res.data
        click.echo("================ Indexing Complete ================")
        click.echo(f"Files Scanned    : {d.get('files_scanned')}")
        click.echo(f"Files Indexed    : {d.get('files_indexed')}")
        click.echo(f"Files Skipped    : {d.get('files_skipped')}")
        click.echo(f"Files Failed     : {d.get('files_failed')}")
        click.echo(f"Duration (s)     : {d.get('duration_seconds')}s")
        click.echo(f"Peak RSS (MB)    : {d.get('peak_rss_mb')} MB")
        click.echo("===================================================")
    else:
        click.echo(f"Indexing failed: {res.error}")


@cli.command("search")
@click.argument("query", required=True)
@click.option("--path-filter", "-p", help="Filter by relative path substring")
@click.option("--type-filter", "-t", help="Filter by file type or category")
@click.option("--limit", "-n", type=int, default=5, help="Number of evidence results to return")
@click.pass_context
def search_cmd(ctx, query, path_filter, type_filter, limit):
    """Performs semantic vector search across indexed evidence."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    search_engine = ctx_objs["search_engine"]
    res = search_engine.search(
        query=query,
        path_filter=path_filter,
        file_type_filter=type_filter,
        limit=limit,
    )

    if not res.evidence:
        click.echo(f"No relevant evidence found for query: '{query}'")
        return

    click.echo(f"Found {len(res.evidence)} evidence items for: '{query}':\n")
    for idx, ev in enumerate(res.evidence, 1):
        click.echo(f"[{idx}] {ev.formatted_citation()}")
        click.echo(f"Artifact Type : {ev.artifact_type}")
        click.echo("Content Snippet:")
        for line in ev.text.splitlines()[:8]:
            click.echo(f"    {line}")
        click.echo("-" * 60)


@cli.command("ask")
@click.argument("query", required=True)
@click.pass_context
def ask_cmd(ctx, query):
    """Asks the MindFS Agent a question grounded in your indexed filesystem."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    agent = ctx_objs["agent"]
    response = agent.ask(query)
    click.echo(response.answer)


@cli.command("rebuild")
@click.pass_context
def rebuild_cmd(ctx):
    """Clears and fully rebuilds the workspace index."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    indexer = ctx_objs["indexer"]
    click.echo("Rebuilding index from scratch...")
    res = indexer.rebuild_index()
    click.echo("Index rebuild completed:")
    click.echo(json.dumps(res, indent=2))


@cli.command("remove")
@click.argument("path", required=True)
@click.pass_context
def remove_cmd(ctx, path):
    """Removes a file or directory from the index."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    indexer = ctx_objs["indexer"]
    resolved = ctx_objs["sandbox"].validate_and_resolve(path)
    if resolved.is_dir():
        count = indexer.remove_directory(path)
        click.echo(f"Removed directory and {count} indexed vectors.")
    else:
        indexer.remove_file(path)
        click.echo(f"Removed file '{path}' from index.")


@cli.command("diagnostics")
@click.pass_context
def diagnostics_cmd(ctx):
    """Displays memory diagnostics and peak RSS measurements."""
    ctx_objs = build_system_context(ctx.obj["config_path"], ctx.obj["workspace_root"])
    resources = ctx_objs["resources"]
    summary = resources.get_summary()
    click.echo("================ MindFS Memory & Diagnostics ================")
    click.echo(f"Current Process RSS : {summary['current_rss_mb']} MB")
    click.echo(f"Peak Process RSS    : {summary['peak_rss_mb']} MB")
    click.echo(f"Budget Limit        : {summary['budget_max_rss_mb']} MB (< 1.7 GB target)")
    click.echo(f"Remaining Margin    : {summary['budget_remaining_mb']} MB")
    click.echo(f"Operations Tracked  : {summary['operations_recorded']}")
    click.echo("=============================================================")


@cli.command("ui")
@click.option("--host", default="0.0.0.0", help="Host interface to bind")
@click.option("--port", default=8765, type=int, help="Port to listen on")
@click.pass_context
def ui_cmd(ctx, host, port):
    """Launches the MindFS interactive Web UI."""
    from mindfs.ui.server import run_ui_server
    run_ui_server(
        host=host,
        port=port,
        config_path=ctx.obj.get("config_path"),
        workspace_root=ctx.obj.get("workspace_root"),
    )


def main():
    cli()


if __name__ == "__main__":
    main()

