"""Read account storage quota through standard manually authorized Colab credentials."""
import json
from pathlib import Path


def quota():
    import google.auth
    from googleapiclient.discovery import build
    credentials,_=google.auth.default()
    value=build("drive","v3",credentials=credentials,cache_discovery=False).about().get(fields="storageQuota").execute()["storageQuota"]
    limit=int(value["limit"]) if value.get("limit") else None
    used=int(value.get("usage",0))
    return {"limit_bytes":limit,"used_bytes":used,"free_bytes":None if limit is None else limit-used}


def require_space(minimum=512*2**20):
    value=quota()
    if value["free_bytes"] is not None and value["free_bytes"] < minimum:
        raise OSError(f"Drive account quota insufficient: {value['free_bytes']/2**30:.2f}GiB free; need {minimum/2**30:.2f}GiB reserve")
    return value
