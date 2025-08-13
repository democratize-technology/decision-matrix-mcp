# MIT License
#
# Copyright (c) 2025 Democratize Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Configuration schema definitions for Decision Matrix MCP.

This module defines the complete configuration schema using Pydantic models
for type safety, validation, and auto-completion support.
"""

from pydantic import BaseModel, Field, field_validator


class ValidationLimitsConfig(BaseModel):
    """Configuration for input validation limits."""

    # Session validation
    max_session_id_length: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum length for session IDs",
    )

    # Content validation
    max_topic_length: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum length for decision topic",
    )
    max_option_name_length: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Maximum length for option names",
    )
    max_criterion_name_length: int = Field(
        default=100,
        ge=5,
        le=500,
        description="Maximum length for criterion names",
    )
    max_description_length: int = Field(
        default=1000,
        ge=50,
        le=5000,
        description="Maximum length for descriptions",
    )

    # Business logic limits
    min_options_required: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Minimum number of options required for decision analysis",
    )
    max_options_allowed: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Maximum number of options allowed per decision",
    )
    max_criteria_allowed: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of criteria allowed per decision",
    )

    # Weight validation
    min_criterion_weight: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Minimum weight value for criteria",
    )
    max_criterion_weight: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum weight value for criteria",
    )

    @field_validator("max_options_allowed")
    @classmethod
    def validate_max_options(cls, v, info):
        """Ensure max options is greater than min options."""
        if info.data:
            min_options = info.data.get("min_options_required", 2)
            if v <= min_options:
                raise ValueError(
                    f"max_options_allowed ({v}) must be greater than min_options_required ({min_options})",
                )
        return v

    @field_validator("max_criterion_weight")
    @classmethod
    def validate_weight_range(cls, v, info):
        """Ensure max weight is greater than min weight."""
        if info.data:
            min_weight = info.data.get("min_criterion_weight", 0.1)
            if v <= min_weight:
                raise ValueError(
                    f"max_criterion_weight ({v}) must be greater than min_criterion_weight ({min_weight})",
                )
        return v


class SessionLimitsConfig(BaseModel):
    """Configuration for session management limits."""

    # Maximum active sessions before LRU eviction kicks in
    max_active_sessions: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Maximum active sessions in memory before LRU eviction",
    )

    # Number of sessions to evict when reaching limit (batch eviction)
    lru_eviction_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of sessions to evict in each LRU cleanup batch",
    )

    # Default session limits for SessionManager
    default_max_sessions: int = Field(
        default=50,
        ge=5,
        le=1000,
        description="Default maximum sessions per SessionManager instance",
    )
    default_session_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 1 hour to 1 week
        description="Default session time-to-live in hours",
    )
    default_cleanup_interval_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,  # 1 minute to 1 day
        description="Default cleanup interval in minutes",
    )

    @field_validator("lru_eviction_batch_size")
    @classmethod
    def validate_batch_size(cls, v, info):
        """Ensure batch size is reasonable compared to max active sessions."""
        if info.data:
            max_active = info.data.get("max_active_sessions", 100)
            if v >= max_active:
                raise ValueError(
                    f"lru_eviction_batch_size ({v}) must be less than max_active_sessions ({max_active})",
                )
        return v

    @field_validator("default_max_sessions")
    @classmethod
    def validate_default_max(cls, v, info):
        """Ensure default max sessions is reasonable compared to max active."""
        if info.data:
            max_active = info.data.get("max_active_sessions", 100)
            if v > max_active:
                raise ValueError(
                    f"default_max_sessions ({v}) should not exceed max_active_sessions ({max_active})",
                )
        return v


class PerformanceLimitsConfig(BaseModel):
    """Configuration for performance and concurrency limits."""

    # Orchestrator retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed evaluations",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Initial delay between retries in seconds (with exponential backoff)",
    )

    # Concurrency limits
    max_concurrent_evaluations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent criterion evaluations",
    )

    # Timeout settings
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Default timeout for LLM requests in seconds",
    )
    cot_timeout_seconds: float = Field(
        default=30.0,
        ge=10.0,
        le=300.0,
        description="Timeout for Chain of Thought evaluation in seconds",
    )
    cot_summary_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Quick timeout for Chain of Thought summary generation",
    )


class BackendLimitsConfig(BaseModel):
    """Configuration for LLM backend-specific limits and settings."""

    # Default model names
    bedrock_model: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Default AWS Bedrock model to use",
    )
    litellm_model: str = Field(default="gpt-4o-mini", description="Default LiteLLM model to use")
    ollama_model: str = Field(default="llama3.1:8b", description="Default Ollama model to use")

    # Backend-specific timeouts
    bedrock_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for AWS Bedrock requests",
    )
    litellm_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for LiteLLM requests",
    )
    ollama_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Timeout for Ollama requests",
    )

    # Temperature settings
    default_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM requests",
    )


class ConfigSchema(BaseModel):
    """Complete configuration schema for Decision Matrix MCP."""

    validation: ValidationLimitsConfig = Field(
        default_factory=ValidationLimitsConfig,
        description="Input validation and business logic limits",
    )

    session: SessionLimitsConfig = Field(
        default_factory=SessionLimitsConfig,
        description="Session management configuration",
    )

    performance: PerformanceLimitsConfig = Field(
        default_factory=PerformanceLimitsConfig,
        description="Performance and concurrency limits",
    )

    backend: BackendLimitsConfig = Field(
        default_factory=BackendLimitsConfig,
        description="LLM backend configuration",
    )

    # Environment-specific settings
    environment: str = Field(
        default="production",
        pattern="^(development|staging|production)$",
        description="Environment mode for configuration profiles",
    )

    debug_mode: bool = Field(default=False, description="Enable debug mode with additional logging")

    # Configuration metadata
    config_version: str = Field(default="1.0.0", description="Configuration schema version")

    class Config:
        """Pydantic configuration."""

        # Allow extra fields for forward compatibility
        extra = "allow"
        # Use enum values for serialization
        use_enum_values = True
        # Validate assignment for runtime configuration updates
        validate_assignment = True
