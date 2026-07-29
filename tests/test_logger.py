"""Tests for src.utils.logger."""

import logging

from src.utils import logger as logger_module


def test_get_logger_returns_logger_instance() -> None:
    result = logger_module.get_logger("test_module")
    assert isinstance(result, logging.Logger)


def test_get_logger_shares_single_configuration() -> None:
    logger_a = logger_module.get_logger("module_a")
    logger_b = logger_module.get_logger("module_b")

    # Both should be children of the same configured root logger,
    # and neither should have attached its own handlers.
    assert logger_a.name == "conformallab.module_a"
    assert logger_b.name == "conformallab.module_b"
    assert logger_a.handlers == []
    assert logger_b.handlers == []


def test_repeated_configure_logging_does_not_duplicate_handlers() -> None:
    root_logger = logging.getLogger("conformallab")
    handler_count_before = len(root_logger.handlers)

    logger_module.configure_logging()
    logger_module.configure_logging()

    assert len(root_logger.handlers) == handler_count_before