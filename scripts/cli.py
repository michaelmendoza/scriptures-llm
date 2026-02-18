"""CLI commands for the RAG system."""

import sys

from rich.console import Console
from rich.panel import Panel

from rag.ingest import ingest_documents
from rag.retriever import retrieve
from rag.generator import generate
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


def cmd_query(question: str):
    """Answer a single question."""
    contexts = retrieve(question)
    display_sources(contexts)

    if not contexts:
        console.print("[yellow]No documents ingested. Run 'python main.py ingest' first.[/yellow]")
        return

    console.print("[bold]Answer:[/bold]")
    generate(question, contexts)


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

        contexts = retrieve(question)
        display_sources(contexts)

        if not contexts:
            console.print("[yellow]No documents ingested. Run 'python main.py ingest' first.[/yellow]")
            continue

        console.print("[bold green]Assistant:[/bold green]")
        generate(question, contexts)

    console.print("\n[dim]Goodbye.[/dim]")


def cmd_download():
    """Download KJV Bible scriptures as test data."""
    from scripts.download_scriptures import main as download_main
    console.print(Panel("Downloading KJV scriptures to [bold]data/[/bold]"))
    download_main()
    console.print("[green]Done.[/green]")


def run_cli():
    if len(sys.argv) < 2:
        console.print("Usage:")
        console.print("  python main.py [bold]download[/bold]           Download KJV Bible test data")
        console.print("  python main.py [bold]ingest[/bold]             Ingest documents from data/")
        console.print('  python main.py [bold]query[/bold] "question"   Ask a one-shot question')
        console.print("  python main.py [bold]chat[/bold]               Interactive chat mode")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "download":
        cmd_download()
    elif command == "ingest":
        cmd_ingest()
    elif command == "query":
        if len(sys.argv) < 3:
            console.print("[red]Please provide a question: python main.py query \"your question\"[/red]")
            sys.exit(1)
        cmd_query(" ".join(sys.argv[2:]))
    elif command == "chat":
        cmd_chat()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)
