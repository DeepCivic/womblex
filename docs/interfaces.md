# Interface descriptions

The `/v1` service API ([service-api.md](service-api.md)) described in published interface
standards, as it is today. Each description is generated from the code
(`src/womblex/api/interface_docs.py`), and `tests/test_interface_docs.py` fails when a
committed file no longer matches what the code serves. Regenerate with:

```bash
uv run python -m womblex.api.interface_docs
```

| Standard | Version | File | Status |
|---|---|---|---|
| OpenAPI | 3.2.0 | [`interfaces/openapi.json`](interfaces/openapi.json) | As is |
| Swagger | 2.0 | [`interfaces/swagger.json`](interfaces/swagger.json) | As is |

## OpenAPI 3.2.0

The document the service generates at `/openapi.json` (FastAPI's OpenAPI 3.1 output), with
two things that output leaves out:

- **Security.** `bearerAuth` (an HTTP bearer service token from the client registry,
  `WOMBLEX_API_CLIENTS`) on every operation except `GET /v1/health` and `GET /v1/ready`,
  which take none. Each operation's requirement names the scope it checks (`read`, `submit`,
  `read_raw` or `admin`), as OpenAPI 3.1 and later allow for a non-OAuth scheme. The scope
  is read from the service's own answer: the generator sends a token holding no scope to
  each operation and records the scope its 403 names. `POST /v1/hooks/{workflow}/{hook}` is
  `hookSignature` instead: the `X-Womblex-Signature` header, `sha256=<hex>` over the raw body.
- **A description** naming the token source and the scopes.

The document declares `"openapi": "3.2.0"`. Nothing in it needs a 3.2 feature; 3.2 keeps
3.1's JSON Schema 2020-12 dialect, so the generated schemas carry over unchanged. It was
checked against the OpenAPI Initiative's 3.2 schema (`schema` and `schema-base`,
2025-09-17) with no errors when written; the test here does not repeat that check, since
the repository has no JSON Schema validator.

Not described:

- **Outbound webhooks.** The events a workflow's webhook destination is sent
  ([workflows.md](workflows.md)) are not routes of this API, so no `webhooks` entry is
  written.
- **Checks inside a handler.** A scope a route checks in its body rather than through its
  dependencies (a document text layer that needs `read_raw`) is in the route's prose in
  [service-api.md](service-api.md), not in its security requirement.
- **Servers.** A deployment's base URL is its own; no `servers` entry is written.

The running service still serves its own 3.1 document at `/openapi.json`, without the
security declarations.

## Swagger 2.0

The OpenAPI 3.2.0 document converted to Swagger 2.0, for tools that take only 2.0:

- `components/schemas` become `definitions`, and every `$ref` follows them.
- A JSON body becomes an `in: body` parameter. The upload's multipart body becomes one
  `in: formData` parameter of `type: file`; Swagger 2.0 has no array of files, so the field
  is described as repeated once per file.
- A nullable value (`anyOf` with `null`) is its type with `x-nullable: true`. A union of two
  types (a validation error's `loc` items) has no 2.0 form and is left untyped, its
  branches kept under `x-anyOf`. `const` becomes a one-value `enum`; a numeric exclusive
  bound becomes `minimum` / `maximum` with the boolean `exclusiveMinimum` / `exclusiveMaximum`.
- Swagger 2.0 has no bearer scheme and allows scopes only on OAuth2, so `bearerAuth` is an
  `apiKey` in the `Authorization` header (`Bearer <token>`), and each operation's scope is
  in `x-womblex-scope`.

It was checked against the archived Swagger 2.0 JSON Schema with no errors when written.
