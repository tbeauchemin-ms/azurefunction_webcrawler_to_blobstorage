# Azure Function Web Crawler to Blob Storage

This project is an **Azure Function** that acts as a robust web crawler, designed to extract content (including HTML, PDF, DOCX, XLSX) from target websites, process it, and upload the results as JSON documents to Azure Blob Storage. It is highly configurable and optimized for use with data pipelines and AI search scenarios.

## Features

- Crawl and extract content from websites (HTML, PDF, DOCX, XLSX)
- Stores extracted content and logs as JSON in Azure Blob Storage
- Supports depth-limited recursive crawling and sitemap parsing
- Skips binary and irrelevant files via regex filtering
- Application Insights support for tracking and monitoring
- Works as a timer-triggered Azure Function (scheduled crawl), with a simple HTTP health check endpoint
- Dockerfile included for easy container deployment

## Project Structure

```text
azurefunction_webcrawler_to_blobstorage/
├── .gitignore
├── Dockerfile
├── README.md
├── crawler.py
├── function_app.py
├── host.json
├── requirements.txt
├── .funcignore
├── .dockerignore
├── appsettings.json         # Not tracked; for Azure deployment only
├── local.settings.json      # Not tracked; for local development only
```


## Getting Started

### Prerequisites

- Python 3.9+
- [Azure Functions Core Tools](https://learn.microsoft.com/azure/azure-functions/functions-run-local)
- [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
- An Azure Subscription with access to Blob Storage
- Docker (optional, for container deployment)

### Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/jdnuckolls/azurefunction_webcrawler_to_blobstorage.git
    cd azurefunction_webcrawler_to_blobstorage
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure settings:**

    - `local.settings.json` is used for local development (not included in repo; create your own based on the sample).
    - Set environment variables for storage accounts, crawl parameters, and Application Insights as needed.

4. **Run locally:**
    ```bash
    func start
    ```

5. **Deploy to Azure:**
    - Deploy using Azure Functions Core Tools or Azure Portal.
    - For Docker deployment, use the included Dockerfile.

### Usage

- The crawler runs on a schedule (default: daily at 2am UTC) via a timer trigger.
- To check health/status, send an HTTP GET to the `/api/ping` endpoint.

## Configuration

Main settings are provided via environment variables or `local.settings.json`:

- `STORAGE_ACCOUNT_NAME`, `CONTAINER_NAME`, `LOG_CONTAINER_NAME`
- `BASE_URLS` (semicolon-separated list of crawl entry points)
- `ALLOW_DOMAINS` (which domains to allow crawling)
- `MAX_DEPTH` (maximum crawl depth)
- `SKIP_REGEXES` (regex patterns for files/links to skip)
- See `appsettings.json` and `local.settings.json` for more.

## Security

- **DO NOT commit secrets** (such as connection strings or API keys) to this repository.
- The `.gitignore` is set up to exclude sensitive config files and local environments.

## License

MIT License (add your license here if different).

## Author

[Jeff Nuckolls](https://github.com/jdnuckolls)

---

*Inspired by best practices for large-scale content ingestion and search.*
