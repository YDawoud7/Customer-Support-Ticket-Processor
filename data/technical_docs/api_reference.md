# API Reference

## Authentication
All API requests require an API key passed in the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```
API keys can be generated from Account Settings > API Keys. Each key has configurable scopes (read, write, admin).

## Rate Limits
| Plan | Requests/Minute | Requests/Day |
|------|----------------|-------------|
| Starter | 60 | 10,000 |
| Professional | 300 | 100,000 |
| Enterprise | 1,000 | Unlimited |

When rate limited, the API returns HTTP 429 with a `Retry-After` header indicating seconds until the next available request.

## Common Endpoints

### GET /api/v1/projects
List all projects for the authenticated user.
- **Query Parameters**: `page` (default: 1), `per_page` (default: 20), `status` (active/archived).
- **Response**: JSON array of project objects.

### POST /api/v1/projects
Create a new project.
- **Body**: `{ "name": "string", "description": "string", "template_id": "string" }`
- **Response**: Created project object with `id`.

### GET /api/v1/projects/{id}/export
Export a project to the specified format.
- **Query Parameters**: `format` (pdf/docx/png), `quality` (draft/standard/high).
- **Response**: Binary file download or async job ID for large exports.

### POST /api/v1/webhooks
Register a webhook for event notifications.
- **Body**: `{ "url": "string", "events": ["project.created", "project.exported", "payment.received"] }`
- **Response**: Webhook object with `id` and `secret` for signature verification.

## Error Responses
All errors follow a standard format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": {}
  }
}
```

Common error codes:
- `AUTH_INVALID`: Invalid or expired API key.
- `RATE_LIMITED`: Too many requests. Check `Retry-After` header.
- `NOT_FOUND`: The requested resource does not exist.
- `VALIDATION_ERROR`: Request body failed validation. Check `details` for field-level errors.
- `INTERNAL_ERROR`: Unexpected server error. Retry with exponential backoff.

## SDKs
Official SDKs are available for Python, JavaScript, and Go. Install via your language's package manager:
- Python: `pip install acme-sdk`
- JavaScript: `npm install @acme/sdk`
- Go: `go get github.com/acme/sdk-go`
