"""Shared model configuration for context-pack value objects."""

from pydantic import BaseModel, ConfigDict


class ContextPackModel(BaseModel):
    """Immutable validated value object."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
