#!/usr/bin/env python3
"""
Fix broken f-strings that are split across lines incorrectly.
"""
import re
from pathlib import Path


def fix_broken_fstring(lines, start_idx):
    """Fix a broken f-string starting at start_idx."""
    # Find the f-string start
    line = lines[start_idx]
    
    # Check if it's an f-string
    if not (line.strip().startswith('f"') or line.strip().startswith("f'")):
        return None, 0
    
    # Find where the f-string starts
    fstring_start = line.find('f"') if 'f"' in line else line.find("f'")
    if fstring_start == -1:
        return None, 0
    
    quote_char = line[fstring_start + 1]  # " or '
    
    # Collect all lines until we find the closing quote
    fstring_lines = [line]
    current_idx = start_idx
    brace_count = 0
    in_string = False
    quote_count = 0
    
    # Parse the first line to see if we're in a string
    for char in line[fstring_start + 2:]:
        if char == quote_char and (not char or line[line.index(char) - 1] != '\\'):
            quote_count += 1
            if quote_count % 2 == 0:
                # String is closed on this line
                return None, 0
        elif char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
    
    # If we get here, the string continues
    current_idx += 1
    while current_idx < len(lines):
        next_line = lines[current_idx]
        fstring_lines.append(next_line)
        
        # Check if this line closes the string
        for char in next_line:
            if char == quote_char:
                quote_count += 1
                if quote_count % 2 == 0:
                    # Found closing quote
                    break
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        if quote_count % 2 == 0:
            break
        
        current_idx += 1
    
    if current_idx >= len(lines):
        return None, 0
    
    # Now reconstruct the f-string on a single line or properly formatted
    # Extract the prefix (indentation and code before f-string)
    prefix = line[:fstring_start].rstrip()
    if prefix and not prefix.endswith('(') and not prefix.endswith('='):
        prefix = prefix.rstrip()
    
    # Extract all the content
    content_parts = []
    for i, fline in enumerate(fstring_lines):
        if i == 0:
            # First line: get content after f" or f'
            content_start = fline.find('f' + quote_char) + 2
            content_parts.append(fline[content_start:].rstrip())
        else:
            # Subsequent lines: get the content (strip leading whitespace that's just indentation)
            stripped = fline.lstrip()
            # Remove trailing quote if present
            if stripped.endswith(quote_char):
                content_parts.append(stripped[:-1])
            else:
                content_parts.append(stripped)
    
    # Join content and fix braces
    full_content = ' '.join(content_parts)
    
    # Remove extra whitespace around braces
    full_content = re.sub(r'\{\s+', '{', full_content)
    full_content = re.sub(r'\s+\}', '}', full_content)
    
    # Reconstruct the line
    # Try to keep it on one line if reasonable, otherwise use parentheses
    if len(prefix + f'f{quote_char}{full_content}{quote_char}') < 120:
        new_line = prefix + f'f{quote_char}{full_content}{quote_char}'
        # Replace the lines
        num_lines = len(fstring_lines)
        return new_line, num_lines
    else:
        # Use parentheses for multi-line
        new_lines = [prefix + f'f{quote_char}{full_content}{quote_char}']
        return new_lines, len(fstring_lines)


def fix_file(filepath):
    """Fix broken f-strings in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        changes_made = False
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line has a broken f-string pattern
            # Look for f" or f' followed by text ending with {
            if re.search(r'f["\'][^"\']*\{[\s]*$', line):
                # Try to fix it
                fixed, num_lines = fix_broken_fstring_simple(lines, i)
                if fixed:
                    fixed_lines.append(fixed)
                    i += num_lines
                    changes_made = True
                    continue
            
            fixed_lines.append(line)
            i += 1
        
        if changes_made:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True
        return False
    
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def fix_broken_fstring_simple(lines, start_idx):
    """Simpler fix: find the f-string and put it on one line."""
    line = lines[start_idx]
    
    # Find f-string start
    fstring_match = re.search(r'f["\']', line)
    if not fstring_match:
        return None, 0
    
    quote_char = line[fstring_match.end() - 1]
    fstring_start = fstring_match.start()
    
    # Get prefix (everything before f-string)
    prefix = line[:fstring_start].rstrip()
    
    # Collect f-string content across lines
    content_parts = []
    current_idx = start_idx
    
    # Get content from first line
    first_content = line[fstring_match.end():].rstrip()
    if first_content.endswith(quote_char):
        # Already closed
        return None, 0
    
    content_parts.append(first_content)
    current_idx += 1
    
    # Collect until we find closing quote
    while current_idx < len(lines):
        next_line = lines[current_idx]
        stripped = next_line.lstrip()
        
        # Check if this line closes the string
        if quote_char in stripped:
            # Find the closing quote
            quote_pos = stripped.find(quote_char)
            # Check if it's escaped
            if quote_pos > 0 and stripped[quote_pos - 1] == '\\':
                # Escaped, continue
                content_parts.append(stripped)
                current_idx += 1
                continue
            
            # Found closing quote
            content_parts.append(stripped[:quote_pos + 1])
            break
        
        content_parts.append(stripped)
        current_idx += 1
    
    if current_idx >= len(lines):
        return None, 0
    
    # Join content, cleaning up whitespace
    full_content = ' '.join(content_parts)
    
    # Clean up whitespace around braces
    full_content = re.sub(r'\{\s+', '{', full_content)
    full_content = re.sub(r'\s+\}', '}', full_content)
    full_content = re.sub(r'\s+', ' ', full_content)
    
    # Reconstruct
    new_line = prefix + f'f{quote_char}{full_content}\n'
    num_lines = current_idx - start_idx + 1
    
    return new_line, num_lines


def main():
    """Main function."""
    corerec_path = Path('corerec')
    fixed_count = 0
    
    for py_file in corerec_path.rglob('*.py'):
        if fix_file(py_file):
            print(f"✓ Fixed: {py_file}")
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()



