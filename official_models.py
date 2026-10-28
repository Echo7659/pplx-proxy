"""Validate Perplexity's website model picker without guessing model IDs."""

import re


_TIERS={"free": 0, "pro": 1, "max": 2}
_PAID_STATUSES={"active", "trialing", "white_glove_past_due"}
_MODEL_ID=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_MODELS=2048
_MAX_SEARCH_OPTIONS=256


def derive_account_type(user):
    """Mirror the website's paid-search subscription gate."""
    if not isinstance(user, dict):
        raise ValueError("Account user must be an object")
    status=user.get("subscription_status")
    tier=user.get("subscription_tier")
    if status is not None and not isinstance(status, str):
        raise ValueError("Account subscription status must be a string")
    if tier is not None and not isinstance(tier, str):
        raise ValueError("Account subscription tier must be a string")
    if status not in _PAID_STATUSES or tier == "viewer":
        return "free"
    return "max" if tier == "max" else "pro"


def _text(value, name, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Invalid official model {name}")
    return value


def _reference(value, name):
    if value is not None and (
        not isinstance(value, str) or not _MODEL_ID.fullmatch(value)
    ):
        raise ValueError(f"Invalid official model {name}")
    return value


def _metadata(models, preference):
    metadata=models.get(preference)
    if not isinstance(metadata, dict):
        raise ValueError("Official search option references missing model metadata")
    _text(metadata.get("label"), "label")
    _text(metadata.get("mode"), "mode", limit=80)
    provider=metadata.get("provider")
    if provider is not None:
        _text(provider, "provider", limit=80)
    return metadata


def parse_catalog(payload, account_type="pro"):
    """Return account-eligible search options from the official v2 picker.

    The metadata registry also contains retired and browser/computer models.
    Only search-picker references with search-mode metadata are exposed. The
    authenticated endpoint supplies audience eligibility; audience is not a
    client-side availability gate. Unknown tiers are omitted conservatively.
    Invalid replacements raise before callers persist or activate any state.
    """
    if not isinstance(account_type, str) or account_type not in _TIERS:
        raise ValueError("Account type must be free, pro, or max")
    if not isinstance(payload, dict) or payload.get("config_schema") != "v2":
        raise ValueError("Unsupported official model configuration schema")
    if payload.get("organization_model_policy") is not None:
        raise ValueError("Unsupported organization model policy; retaining previous catalog")
    if payload.get("fast_model") or payload.get("speed_options"):
        raise ValueError("Unsupported official fast/speed model variants; retaining previous catalog")

    registry=payload.get("models")
    options=payload.get("search_config")
    defaults=payload.get("default_models")
    if not isinstance(registry, dict) or not 0 < len(registry) <= _MAX_MODELS:
        raise ValueError("Invalid official model metadata registry")
    if not isinstance(options, list) or not 0 < len(options) <= _MAX_SEARCH_OPTIONS:
        raise ValueError("Invalid official search model options")
    if not isinstance(defaults, dict):
        raise ValueError("Missing official default models")
    default=_reference(defaults.get("search"), "default preference")
    if default is None:
        raise ValueError("Missing official search default")
    default_metadata=_metadata(registry, default)
    if default_metadata["mode"] != "search":
        raise ValueError("Official search default is not a search model")

    result={}
    search_options=0
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("Official search model option must be an object")
        if option.get("fast_model") or option.get("speed_options"):
            raise ValueError("Unsupported official fast/speed model variants; retaining previous catalog")
        tier=option.get("subscription_tier")
        if not isinstance(tier, str):
            raise ValueError("Official search model option is missing its subscription tier")
        if tier not in _TIERS:
            continue
        non_reasoning=_reference(option.get("non_reasoning_model"), "base preference")
        reasoning=_reference(option.get("reasoning_model"), "thinking preference")
        preference=non_reasoning or reasoning
        if preference is None:
            raise ValueError("Official search model option has no model reference")
        metadata=_metadata(registry, preference)
        thinking_metadata=_metadata(registry, reasoning) if reasoning else None
        if metadata["mode"] != "search":
            continue
        if thinking_metadata and thinking_metadata["mode"] != "search":
            raise ValueError("Official thinking preference is not a search model")
        label=_text(option.get("label"), "picker label")
        text_only=option.get("text_only_model", False)
        if not isinstance(text_only, bool):
            raise ValueError("Official text-only marker must be a boolean")
        search_options += 1
        if _TIERS[tier] > _TIERS[account_type]:
            continue
        entry={
            "label": label,
            "provider": metadata.get("provider"),
            "mode": "pro",
            "pref": preference,
            "thinking_pref": reasoning,
            "tier": tier,
            "text_only": text_only,
        }
        if preference in result and result[preference] != entry:
            raise ValueError("Conflicting official search options for the same model")
        result[preference]=entry

    if not search_options:
        raise ValueError("Official picker contains no supported search options")
    if "auto" in result:
        raise ValueError("Official model ID conflicts with the automatic alias")
    auto={
        "label": default_metadata["label"],
        "provider": default_metadata.get("provider"),
        "mode": "pro",
        "pref": default,
        "thinking_pref": None,
        "tier": "free",
        "text_only": False,
    }
    return {"models": {"auto": auto, **result}, "default": default}
