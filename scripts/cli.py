"""CLI commands for the RAG system."""

import sys

from rich.console import Console
from rich.panel import Panel

from rag.ingest import ingest_documents
from rag.retriever import retrieve
from rag.generator import generate, analyze_and_extract, generate_from_facts
from rag import config

console = Console()


def cmd_ingest():
    """Ingest all documents from the data directory."""
    console.print(Panel(f"Ingesting documents from [bold]{config.DATA_DIR}/[/bold]"))
    ingest_documents()
    console.print("[green]Done.[/green]")


def display_sources(contexts: list[dict]):
    """Display retrieved sources."""
    if not contexts:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return
    console.print("\n[dim]Retrieved sources:[/dim]")
    seen = set()
    for ctx in contexts:
        if ctx["source"] not in seen:
            seen.add(ctx["source"])
            console.print(f"  [dim]- {ctx['source']}[/dim]")
    console.print()


def _maybe_reformulate(question: str) -> tuple[str, str]:
    """Optionally reformulate the query for better retrieval.

    Returns ``(search_query, original_query)``.  When reformulation is
    disabled, both values are the original question.
    """
    if config.ENABLE_QUERY_REFORMULATION:
        from rag.query import reformulate_query
        search_query = reformulate_query(question)
        if search_query != question:
            console.print(f"[dim]Reformulated query: {search_query}[/dim]")
        return search_query, question
    return question, question


def _run_pipeline(question: str):
    """Shared query pipeline used by both ``cmd_query`` and ``cmd_chat``."""
    search_query, original_query = _maybe_reformulate(question)

    contexts = retrieve(search_query)
    display_sources(contexts)

    if not contexts:
        console.print("[yellow]No documents ingested. Run 'python main.py ingest' first.[/yellow]")
        return

    if config.ENABLE_PROMPT_CHAINING:
        console.print("[dim]Extracting key facts...[/dim]")
        facts = analyze_and_extract(original_query, contexts)
        console.print("[bold]Answer:[/bold]")
        generate_from_facts(original_query, facts)
    else:
        console.print("[bold]Answer:[/bold]")
        generate(original_query, contexts)


def cmd_query(question: str):
    """Answer a single question."""
    _run_pipeline(question)


def cmd_chat():
    """Interactive chat loop."""
    console.print(Panel("Interactive RAG Chat\nType [bold]quit[/bold] or [bold]exit[/bold] to stop."))

    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        console.print("[bold green]Assistant:[/bold green]")
        _run_pipeline(question)

    console.print("\n[dim]Goodbye.[/dim]")


def cmd_download():
    """Download KJV Bible scriptures as test data."""
    from scripts.download_scriptures import main as download_main
    console.print(Panel("Downloading KJV scriptures to [bold]data/[/bold]"))
    download_main()
    console.print("[green]Done.[/green]")


def cmd_generate_metadata(force: bool = False):
    """Generate LLM-powered chapter and book summaries."""
    from scripts.generate_metadata import main as gen_meta_main
    console.print(Panel("Generating metadata with LLM summaries (two-pass)"))
    gen_meta_main(force=force)
    console.print("[green]Done.[/green]")


def cmd_eval():
    """Run retrieval evaluation against the test set."""
    from rag.evaluation import run_retrieval_eval
    console.print(Panel("Running retrieval evaluation"))
    run_retrieval_eval()
    console.print("[green]Done.[/green]")


def run_cli():
    if len(sys.argv) < 2:
        console.print("Usage:")
        console.print("  python main.py [bold]download[/bold]                    Download KJV Bible test data")
        console.print("  python main.py [bold]generate-metadata[/bold] [--force]  Generate chapter & book summaries via LLM")
        console.print("  python main.py [bold]ingest[/bold]                      Ingest documents from data/")
        console.print('  python main.py [bold]query[/bold] "question"            Ask a one-shot question')
        console.print("  python main.py [bold]chat[/bold]                        Interactive chat mode")
        console.print("  python main.py [bold]eval[/bold]                        Run retrieval evaluation")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "download":
        cmd_download()
    elif command == "generate-metadata":
        force = "--force" in sys.argv[2:]
        cmd_generate_metadata(force=force)
    elif command == "ingest":
        cmd_ingest()
    elif command == "query":
        if len(sys.argv) < 3:
            console.print("[red]Please provide a question: python main.py query \"your question\"[/red]")
            sys.exit(1)
        cmd_query(" ".join(sys.argv[2:]))
    elif command == "chat":
        cmd_chat()
    elif command == "eval":
        cmd_eval()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)
