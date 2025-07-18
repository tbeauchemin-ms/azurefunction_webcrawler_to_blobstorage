# Environment Variables Reference

This document provides a complete reference for all environment variables used by the crawler, organized by use-case.

## 🕸️ Crawl Configuration

Controls the basic crawling behavior and scope.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BASE_URLS` | string | `""` | Semicolon-separated list of URLs to start crawling from |
| `ALLOW_DOMAINS` | string | `""` | Semicolon-separated list of domains to crawl (subdomain support) |
| `SKIP_REGEXES` | string | `""` | Semicolon-separated regex patterns for URLs to skip |
| `MAX_DEPTH` | int | `3` | Maximum crawling depth from seed URLs |
| `MAX_WORKERS` | int | `4` | Number of concurrent crawler workers |
| `USER_AGENT` | string | `Mozilla/5.0 (GenericCrawler/1.0)` | User agent string for HTTP requests |
| `RESPECT_ROBOTS` | bool | `true` | Whether to respect robots.txt files |
| `SAVE_404` | bool | `false` | Whether to save 404 error pages |

### Examples
```bash
BASE_URLS=https://example.com;https://docs.example.com
ALLOW_DOMAINS=example.com;docs.example.com
SKIP_REGEXES=.*\.pdf$;.*login.*
MAX_DEPTH=5
MAX_WORKERS=8
```

## 🌐 Network & Timing Configuration

Controls request timing, retries, and timeout behavior.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REQUEST_DELAY` | float | `0.5` | Delay between requests in seconds (rate limiting) |
| `PAGE_TIMEOUT_MS` | int | `45000` | Page load timeout in milliseconds |
| `NETWORK_IDLE_WAIT_MS` | int | `0` | Time to wait for network idle after page load |
| `RETRY_COUNT` | int | `3` | Number of retry attempts for failed requests |
| `RETRY_BACKOFF` | float | `2.0` | Backoff factor for retry delays |

### Examples
```bash
REQUEST_DELAY=1.0
PAGE_TIMEOUT_MS=60000
RETRY_COUNT=5
RETRY_BACKOFF=1.5
```

## 📄 Content Processing Configuration

Controls how different content types are processed and chunked.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_CONTENT_CHARS` | int | `500000` | Maximum characters to extract from a single page |
| `INCLUDE_PDFS` | bool | `true` | Whether to process PDF files |
| `INCLUDE_DOCX` | bool | `true` | Whether to process DOCX files |
| `INCLUDE_XLSX` | bool | `true` | Whether to process XLSX files |
| `PDF_DOWNLOAD_TIMEOUT_MS` | int | `60000` | Timeout for PDF downloads in milliseconds |
| `PDF_CHUNK_CHARS` | int | `4000` | Character limit for PDF text chunks |
| `PDF_CHUNK_OVERLAP` | int | `300` | Character overlap between PDF chunks |
| `MAX_PDF_BYTES` | int | `20000000` | Maximum PDF file size to process (20MB) |

### Examples
```bash
MAX_CONTENT_CHARS=1000000
INCLUDE_PDFS=true
INCLUDE_DOCX=false
INCLUDE_XLSX=false
MAX_PDF_BYTES=50000000
```

## 🗄️ Azure Storage Configuration

Controls Azure Blob Storage connection and authentication.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STORAGE_ACCOUNT_NAME` | ✅ Yes | - | Name of the Azure Storage Account |
| `STORAGE_AUTH_METHOD` | No | `managedidentity` | Auth method: `connectionstring` or `managedidentity` |
| `STORAGE_CONNECTION_STRING` | Conditional | - | Required if using `connectionstring` method |
| `STORAGE_CLIENT_ID` | No | - | Client ID for user-assigned managed identity |
| `CONTAINER_NAME` | No | `content` | Container name for storing crawled content |
| `LOG_CONTAINER_NAME` | No | `logs` | Container name for storing logs |

### Examples
```bash
# Using managed identity
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=managedidentity
STORAGE_CLIENT_ID=12345678-1234-1234-1234-123456789012
CONTAINER_NAME=webcrawl-content
LOG_CONTAINER_NAME=webcrawl-logs

# Using connection string
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=connectionstring
STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

## 🤖 Azure OpenAI Configuration

Controls Azure OpenAI connection, authentication, and embedding settings.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | ✅ Yes | - | Azure OpenAI service endpoint |
| `AZURE_OPENAI_AUTH_METHOD` | No | `managedidentity` | Auth method: `apikey` or `managedidentity` |
| `AZURE_OPENAI_API_KEY` | Conditional | - | Required if using `apikey` method |
| `AZURE_OPENAI_CLIENT_ID` | No | - | Client ID for user-assigned managed identity |
| `AZURE_OPENAI_API_VERSION` | No | `2023-05-15` | API version to use |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | ✅ Yes | `myproject-text-embedding-ada-002` | Custom deployment name |
| `AZURE_OPENAI_EMBEDDING_MODEL_NAME` | No | `text-embedding-ada-002` | Actual model name |
| `EMBEDDING_TOKEN_LIMIT` | No | `8191` | Maximum tokens per embedding request |
| `TOKEN_OVERLAP` | No | `300` | Token overlap between text chunks |

### Examples
```bash
# Using API key
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=apikey
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
EMBEDDING_TOKEN_LIMIT=8191

# Using managed identity
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=managedidentity
AZURE_OPENAI_CLIENT_ID=87654321-4321-4321-4321-210987654321
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=my-embedding-deployment
```

## 📊 Monitoring & Logging Configuration

Controls application insights and logging behavior.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APPINSIGHTS_INSTRUMENTATIONKEY` | string | `""` | Azure Application Insights instrumentation key |

### Examples
```bash
APPINSIGHTS_INSTRUMENTATIONKEY=12345678-1234-1234-1234-123456789012
```

## 🏗️ Complete Configuration Examples

### Example 1: Development Environment
```bash
# Crawl Configuration
BASE_URLS=https://docs.example.com
ALLOW_DOMAINS=docs.example.com;example.com
MAX_DEPTH=2
MAX_WORKERS=2

# Network & Timing
REQUEST_DELAY=1.0
PAGE_TIMEOUT_MS=30000
RETRY_COUNT=2

# Content Processing
MAX_CONTENT_CHARS=100000
INCLUDE_PDFS=false
INCLUDE_DOCX=false
INCLUDE_XLSX=false

# Storage (connection string for dev)
STORAGE_ACCOUNT_NAME=devstorageaccount
STORAGE_AUTH_METHOD=connectionstring
STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# OpenAI (API key for dev)
AZURE_OPENAI_ENDPOINT=https://dev-openai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=apikey
AZURE_OPENAI_API_KEY=dev-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
```

### Example 2: Production Environment
```bash
# Crawl Configuration
BASE_URLS=https://company.com;https://docs.company.com;https://support.company.com
ALLOW_DOMAINS=company.com;docs.company.com;support.company.com
MAX_DEPTH=5
MAX_WORKERS=8
RESPECT_ROBOTS=true

# Network & Timing
REQUEST_DELAY=0.5
PAGE_TIMEOUT_MS=45000
RETRY_COUNT=3
RETRY_BACKOFF=2.0

# Content Processing
MAX_CONTENT_CHARS=500000
INCLUDE_PDFS=true
INCLUDE_DOCX=true
INCLUDE_XLSX=true
MAX_PDF_BYTES=20000000

# Storage (managed identity)
STORAGE_ACCOUNT_NAME=prodstorageaccount
STORAGE_AUTH_METHOD=managedidentity
STORAGE_CLIENT_ID=12345678-1234-1234-1234-123456789012
CONTAINER_NAME=webcrawl-content
LOG_CONTAINER_NAME=webcrawl-logs

# OpenAI (managed identity)
AZURE_OPENAI_ENDPOINT=https://prod-openai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=managedidentity
AZURE_OPENAI_CLIENT_ID=87654321-4321-4321-4321-210987654321
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=prod-text-embedding-ada-002
EMBEDDING_TOKEN_LIMIT=8191
TOKEN_OVERLAP=300

# Monitoring
APPINSIGHTS_INSTRUMENTATIONKEY=prod-insights-key
```

### Example 3: High-Volume Crawling
```bash
# Crawl Configuration
BASE_URLS=https://large-site.com
ALLOW_DOMAINS=large-site.com
MAX_DEPTH=10
MAX_WORKERS=16
USER_AGENT=MyCompany-Crawler/2.0

# Network & Timing (aggressive settings)
REQUEST_DELAY=0.1
PAGE_TIMEOUT_MS=30000
NETWORK_IDLE_WAIT_MS=2000
RETRY_COUNT=5
RETRY_BACKOFF=1.5

# Content Processing (optimized)
MAX_CONTENT_CHARS=1000000
INCLUDE_PDFS=true
INCLUDE_DOCX=false
INCLUDE_XLSX=false
PDF_DOWNLOAD_TIMEOUT_MS=120000
MAX_PDF_BYTES=50000000

# Storage & OpenAI (same as production example)
```

## 🔧 Environment Variable Priority

1. **Required Variables**: Must be set or the application will fail
2. **Conditional Variables**: Required based on other settings (e.g., auth method)
3. **Optional Variables**: Have sensible defaults

## 🚨 Security Best Practices

1. **Never hardcode secrets**: Use environment variables for all sensitive data
2. **Use managed identity**: Prefer managed identity over API keys in production
3. **Least privilege**: Grant only necessary permissions to managed identities
4. **Environment separation**: Use different configurations for dev/test/prod
5. **Monitor access**: Enable logging and monitoring for all Azure resources

## 🐛 Troubleshooting

### Common Configuration Issues

1. **Missing required variables**: Check all variables marked as "Required"
2. **Authentication failures**: Verify auth method and credentials are correct
3. **Network timeouts**: Adjust timeout values for slow networks
4. **Rate limiting**: Increase `REQUEST_DELAY` if getting rate limited
5. **Memory issues**: Reduce `MAX_WORKERS` or `MAX_CONTENT_CHARS` for large crawls
