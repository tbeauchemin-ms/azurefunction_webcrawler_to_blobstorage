# Authentication Configuration Guide

This document explains the updated authentication configuration for both Azure Blob Storage and Azure OpenAI services.

## Overview

The crawler now supports multiple authentication methods for both Azure Blob Storage and Azure OpenAI, with clear separation of configuration and improved security practices.

## Azure Blob Storage Authentication

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STORAGE_ACCOUNT_NAME` | ✅ Yes | - | Name of the Azure Storage Account |
| `STORAGE_AUTH_METHOD` | No | `managedidentity` | Authentication method: `connectionstring` or `managedidentity` |
| `STORAGE_CONNECTION_STRING` | Conditional | - | Required if using `connectionstring` method |
| `STORAGE_CLIENT_ID` | No | - | Client ID for user-assigned managed identity |

### Authentication Methods

#### 1. Connection String Authentication
```bash
STORAGE_AUTH_METHOD=connectionstring
STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

#### 2. System-Assigned Managed Identity
```bash
STORAGE_AUTH_METHOD=managedidentity
# STORAGE_CLIENT_ID not set or empty
```

#### 3. User-Assigned Managed Identity
```bash
STORAGE_AUTH_METHOD=managedidentity
STORAGE_CLIENT_ID=12345678-1234-1234-1234-123456789012
```

## Azure OpenAI Authentication

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | ✅ Yes | - | Azure OpenAI service endpoint |
| `AZURE_OPENAI_AUTH_METHOD` | No | `managedidentity` | Authentication method: `apikey` or `managedidentity` |
| `AZURE_OPENAI_API_KEY` | Conditional | - | Required if using `apikey` method |
| `AZURE_OPENAI_CLIENT_ID` | No | - | Client ID for user-assigned managed identity |
| `AZURE_OPENAI_API_VERSION` | No | `2023-05-15` | API version to use |

### Authentication Methods

#### 1. API Key Authentication
```bash
AZURE_OPENAI_AUTH_METHOD=apikey
AZURE_OPENAI_API_KEY=your-api-key-here
```

#### 2. System-Assigned Managed Identity
```bash
AZURE_OPENAI_AUTH_METHOD=managedidentity
# AZURE_OPENAI_CLIENT_ID not set or empty
```

#### 3. User-Assigned Managed Identity
```bash
AZURE_OPENAI_AUTH_METHOD=managedidentity
AZURE_OPENAI_CLIENT_ID=12345678-1234-1234-1234-123456789012
```

## Configuration Examples

### Example 1: Both services using system-assigned managed identity
```bash
# Storage
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=managedidentity

# OpenAI
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=managedidentity
```

### Example 2: Both services using the same user-assigned managed identity
```bash
# Storage
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=managedidentity
STORAGE_CLIENT_ID=12345678-1234-1234-1234-123456789012

# OpenAI (will inherit STORAGE_CLIENT_ID by default)
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=managedidentity
```

### Example 3: Different authentication methods for each service
```bash
# Storage with connection string
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=connectionstring
STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# OpenAI with API key
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=apikey
AZURE_OPENAI_API_KEY=your-api-key-here
```

### Example 4: Different user-assigned managed identities for each service
```bash
# Storage
STORAGE_ACCOUNT_NAME=mystorageaccount
STORAGE_AUTH_METHOD=managedidentity
STORAGE_CLIENT_ID=12345678-1234-1234-1234-123456789012

# OpenAI with different managed identity
AZURE_OPENAI_ENDPOINT=https://myopenai.openai.azure.com/
AZURE_OPENAI_AUTH_METHOD=managedidentity
AZURE_OPENAI_CLIENT_ID=87654321-4321-4321-4321-210987654321
```

## Security Best Practices

1. **Prefer Managed Identity**: Use managed identity over connection strings/API keys when possible
2. **Least Privilege**: Grant only the minimum required permissions to managed identities
3. **User-Assigned vs System-Assigned**: Use user-assigned managed identities for better control and reusability
4. **Environment Separation**: Use different managed identities for different environments (dev/test/prod)

## Required Azure RBAC Permissions

### For Azure Blob Storage
- **Storage Blob Data Contributor** role on the storage account or container
- Alternatively: **Storage Blob Data Reader** + **Storage Blob Data Writer** roles

### For Azure OpenAI
- **Cognitive Services OpenAI User** role on the Azure OpenAI resource
- Alternatively: **Cognitive Services User** role (broader permissions)

## Troubleshooting

### Common Issues

1. **"ManagedIdentityCredential authentication failed"**
   - Ensure the managed identity is enabled on the compute resource
   - Verify RBAC permissions are correctly assigned
   - Check the client ID is correct (if using user-assigned)

2. **"Connection string authentication failed"**
   - Verify the connection string format is correct
   - Ensure the storage account exists and is accessible

3. **"API key authentication failed"**
   - Verify the API key is valid and not expired
   - Check the OpenAI endpoint is correct

### Debugging Tips

- Enable verbose logging to see which authentication method is being used
- Use Azure CLI to test managed identity: `az account get-access-token --resource https://storage.azure.com/`
- Verify resource endpoints and names are correct

## Migration from Previous Configuration

The updated configuration maintains backward compatibility:

- `STORAGE_CREDENTIAL` is now `STORAGE_AUTH_METHOD`
- `STORAGE_CLIENT_ID` continues to work as before
- `USER_ASSIGNED_CLIENT_ID` is maintained for backward compatibility

To migrate, simply update your environment variables to use the new naming convention while keeping the same values.
