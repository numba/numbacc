"""
MLIR Utilities - Common functionality for type name encoding/decoding and MLIR operations.

This module provides utilities to avoid code duplication between different backends
and the frontend type system.
"""

import base64
import re


def encode_type_name(name: str) -> str:
    """
    Encode a type name using base64 URL-safe encoding.

    Args:
        name: The type name to encode

    Returns:
        Base64 encoded string
    """
    return base64.urlsafe_b64encode(name.encode()).decode()


def decode_type_name(encoded: str) -> str:
    """
    Decode a base64 URL-safe encoded type name.

    Args:
        encoded: The base64 encoded string

    Returns:
        Decoded type name
    """
    decoded = base64.urlsafe_b64decode(str(encoded).encode()).decode()
    return decoded


def encode_asm_operation(fqn_parts: list[str]) -> str:
    """
    Encode ASM operation parts into a base64 URL-safe string.

    Args:
        fqn_parts: List of FQN parts to join and encode

    Returns:
        Base64 encoded string
    """
    return base64.urlsafe_b64encode(("$".join(fqn_parts)).encode()).decode()


def decode_asm_operation(encoded: str) -> str:
    """
    Decode a base64 URL-safe encoded ASM operation.

    Args:
        encoded: The base64 encoded string

    Returns:
        Decoded operation string
    """
    return base64.urlsafe_b64decode(encoded.encode()).decode()


def parse_composite_type(tyname: str) -> list[str] | None:
    """
    Parse a composite type name that contains multiple type components.

    Composite types are encoded as "multivalues$type1|type2|type3|..."

    Args:
        tyname: The type name to parse

    Returns:
        List of individual type component names if it's a composite type,
        None if it's not a composite type
    """
    if not tyname.startswith("multivalues$"):
        return None

    _, _, raw_items = tyname.partition("$")
    items = raw_items.split("|")
    return items


def create_mlir_type_fqn(formatted_name: str):
    """
    Create an FQN for MLIR types with proper encoding.

    Args:
        formatted_name: The formatted type name

    Returns:
        FQN object with encoded qualifiers
    """
    from spy.fqn import FQN

    if formatted_name == "()":
        return FQN(["mlir", "type", "()"])
    else:
        humane_name = "_" + re.sub(r"[^a-zA-Z0-9_]", "", formatted_name)
        assert humane_name, formatted_name
        full_name = encode_type_name(formatted_name)

        return FQN(["mlir", "type", humane_name]).with_qualifiers([full_name])
