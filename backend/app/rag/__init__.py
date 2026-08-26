def collection_for(repo: str) -> str:
    return f"{repo.replace('/', '__')}_audit"
