import os
# files larger than 100mb will be added to .gitignore after running this script
# Function to find files greater than 100MB
def find_large_files(directory, size_limit_mb=100):
    large_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > size_limit_mb * 1024 * 1024:
                large_files.append(file_path)
    return large_files

# Directory to search
search_directory = '.'

# Find large files
large_files = find_large_files(search_directory)

# Write to .gitignore
with open('.gitignore', 'a') as gitignore:
    for file in large_files:
        gitignore.write(f"{file}\n")

print("Large files added to .gitignore")