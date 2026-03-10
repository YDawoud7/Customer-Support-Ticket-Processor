# Troubleshooting Guide

## Application Crashes on Startup
1. **Check System Requirements**: Ensure your system meets the minimum requirements (8GB RAM, 2GB disk space, OS version 12+).
2. **Update the Application**: Download the latest version from the releases page.
3. **Clear Application Data**: Delete the local cache at `~/.appdata/cache/` and restart.
4. **Check Logs**: Review the crash log at `~/.appdata/logs/crash.log` for specific error codes.
5. **Reinstall**: If none of the above resolves the issue, perform a clean uninstall and reinstall.

## Slow Performance
1. **Close Unused Projects**: Each open project consumes memory. Keep only active projects open.
2. **Disable Unused Plugins**: Go to Settings > Plugins and disable any plugins you are not actively using.
3. **Check Resource Usage**: Open the built-in performance monitor (Ctrl+Shift+P) to identify bottlenecks.
4. **Increase Memory Allocation**: In Settings > Advanced, increase the max memory allocation (default: 2GB, recommended: 4GB).
5. **Clear Cache**: Large caches can slow down the application. Clear via Settings > Advanced > Clear Cache.

## PDF Export Issues
PDF export problems are among the most commonly reported issues. Common fixes:
1. **Error 0x8007**: See Error Codes Reference for detailed resolution steps.
2. **Blank PDF Output**: Ensure the document is not empty and all layers are visible.
3. **Missing Fonts**: Install any custom fonts used in the document on your system.
4. **Large File Size**: Enable PDF compression in Settings > Export > PDF Options.
5. **Corrupted Output**: Try exporting a single page first to isolate the problematic content.

## Sync Conflicts
When changes from multiple devices conflict:
1. The most recent change is preserved by default.
2. Conflicting versions are saved as separate copies with a "(conflict)" suffix.
3. Open both versions side-by-side to manually merge changes.
4. Delete the conflict copy after resolving differences.

## Login Issues
1. **Forgot Password**: Use the "Forgot Password" link on the login page. Reset link is valid for 24 hours.
2. **Account Locked**: After 5 failed attempts, accounts are locked for 30 minutes. Contact support for immediate unlock.
3. **Two-Factor Authentication**: If you lost your 2FA device, use one of your recovery codes. Contact support if no recovery codes are available.
