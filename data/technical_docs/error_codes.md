# Error Codes Reference

## Application Error Codes

### 0x8007 - Export Module Failure
- **Description**: The export module encountered a fatal error during file conversion.
- **Common Causes**:
  - Corrupted document template.
  - Insufficient disk space for temporary export files.
  - Incompatible font or embedded resource in the source document.
- **Resolution**:
  1. Clear the export cache: Settings > Advanced > Clear Export Cache.
  2. Verify at least 500MB of free disk space is available.
  3. Try exporting to a different format (e.g., DOCX instead of PDF).
  4. If the issue persists, re-import the source document and attempt export again.

### 0x8012 - Authentication Token Expired
- **Description**: The user's session token has expired and needs to be refreshed.
- **Common Causes**:
  - Session inactive for more than 60 minutes.
  - Clock skew between client and server.
- **Resolution**:
  1. Log out and log back in to refresh the token.
  2. If the error persists, clear browser cookies and cache.
  3. Check that your system clock is set to automatic.

### 0x9001 - Database Connection Timeout
- **Description**: The application could not establish a connection to the database within the timeout period.
- **Common Causes**:
  - Network connectivity issues.
  - Database server under heavy load.
  - Firewall blocking the connection port (default: 5432).
- **Resolution**:
  1. Check network connectivity with a simple ping test.
  2. Verify the database server status on the status page.
  3. Ensure port 5432 is open in your firewall settings.
  4. If using a VPN, try disconnecting and reconnecting.

### 0x9015 - File Upload Size Exceeded
- **Description**: The uploaded file exceeds the maximum allowed size.
- **Limits**: Free tier: 25MB, Professional: 100MB, Enterprise: 500MB.
- **Resolution**:
  1. Compress the file before uploading.
  2. Split large files into smaller parts.
  3. Upgrade your plan for higher upload limits.

### 0xA001 - Plugin Compatibility Error
- **Description**: An installed plugin is not compatible with the current application version.
- **Resolution**:
  1. Check the plugin's documentation for version compatibility.
  2. Update the plugin to the latest version.
  3. Disable the plugin temporarily to restore functionality.
  4. Contact the plugin developer if no compatible version exists.
