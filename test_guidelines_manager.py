import os
from guidelines_manager import GuidelinesManager

# Set environment variables for testing
os.environ['REFRESH_INTERVAL_DAYS'] = '3'  # Set a different refresh interval for testing

# Initialize the manager
manager = GuidelinesManager()

# Check if guidelines exist
print(f"Guidelines exist: {manager.guidelines_exist()}")

# Test refresh logic
print(f"Guidelines need refresh: {manager.needs_refresh()}")

# Attempt to update guidelines
if manager.needs_refresh():
    print("Attempting to update guidelines...")
    result = manager.update_guidelines()
    print(f"Update result: {result}")

# Check if guidelines exist after update attempt
print(f"Guidelines exist after update: {manager.guidelines_exist()}")

# Print the refresh interval
print(f"Refresh interval: {manager.refresh_interval_days} days") 