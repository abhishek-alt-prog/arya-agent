"""
Arya Agent — CLI entry point.

Commands:
  setup    — First-run: create courses + initial lessons
  adapt    — Run the nightly adaptation loop
  status   — Check connectivity and progress summary
  dashboard — Launch the Streamlit parent dashboard
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .agent_service import AgentService
from .bff_client import BFFClient
from .config import DEFAULT_CHILD_ID
from .ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arya-agent")


def _get_service() -> AgentService:
    bff = BFFClient()
    ollama = OllamaClient()
    return AgentService(bff, ollama)


def cmd_setup(args: argparse.Namespace) -> None:
    """First-run: generate courses and initial lessons."""
    child_id = args.child_id or DEFAULT_CHILD_ID
    if not child_id:
        logger.error("No child ID provided. Use --child-id or set DEFAULT_CHILD_ID in .env")
        sys.exit(1)

    service = _get_service()
    logger.info("🦉 Starting initial setup for child %s …", child_id)
    result = service.initial_setup(child_id)
    logger.info("✅ Setup complete!")
    print(json.dumps(result, indent=2))


def cmd_adapt(args: argparse.Namespace) -> None:
    """Run the nightly adaptation loop."""
    child_id = args.child_id or DEFAULT_CHILD_ID
    if not child_id:
        logger.error("No child ID provided. Use --child-id or set DEFAULT_CHILD_ID in .env")
        sys.exit(1)

    service = _get_service()
    logger.info("🦉 Starting adaptation run for child %s …", child_id)
    result = service.run_adaptation(child_id)
    logger.info("✅ Adaptation complete!")
    print(json.dumps(result, indent=2, default=str))


def cmd_status(args: argparse.Namespace) -> None:
    """Check connectivity and show progress summary."""
    child_id = args.child_id or DEFAULT_CHILD_ID
    if not child_id:
        logger.error("No child ID provided. Use --child-id or set DEFAULT_CHILD_ID in .env")
        sys.exit(1)

    service = _get_service()
    status = service.get_status(child_id)
    print(json.dumps(status, indent=2))


def cmd_dashboard(_args: argparse.Namespace) -> None:
    """Launch the Streamlit parent dashboard."""
    import subprocess
    import os

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    logger.info("🦉 Launching parent dashboard …")
    subprocess.run(["streamlit", "run", dashboard_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arya-agent",
        description="🦉 Arya's AI Tutor Agent",
    )
    parser.add_argument(
        "--child-id", "-c",
        help="Child ID (overrides DEFAULT_CHILD_ID from .env)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("setup", help="First-run: create courses + initial lessons")
    sub.add_parser("adapt", help="Run the nightly adaptation loop")
    sub.add_parser("status", help="Check connectivity and progress summary")
    sub.add_parser("dashboard", help="Launch the Streamlit parent dashboard")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "adapt":
        cmd_adapt(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
