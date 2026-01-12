# Troubleshooting

## LamAPI configuration

Set these environment variables for LamAPI access:

- `ENTITY_RETRIEVAL_ENDPOINT` (required)
- `ENTITY_RETRIEVAL_TOKEN` (optional)
  - The client sends the token as a `token` query parameter by default.
  - Override with `LAMAPI_TOKEN_PARAM` or disable with `LAMAPI_TOKEN_IN_QUERY=0`.

Use offline mode when working with cached responses only:

```bash
export LAMAPI_OFFLINE=1
export LAMAPI_CACHE_DIR=tests/fixtures/lamapi_cache
```

## Forcing candidates

- `--force-gt-candidate` builds forced ids from the dataset GT.
- `--force-id Q1234` can be repeated to add extra ids to every cell.

For LionLinker, forcing uses mention strings exactly as read from the CSV. If LionLinker applies
mention normalization, the runner attempts to re-use the same normalizer before building the map.

## Native dependencies

Crocodile and Alligator runners attempt to use native libraries if available. If the import fails
or the API does not match, the runners fall back to a LamAPI top-1 baseline and log a warning.
Install the native packages to enable the native pipelines.
