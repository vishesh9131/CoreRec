"""
Fix syntax errors introduced by add_validation_to_engines.py script.

The bug inserted validation code inside function signatures, which caused syntax errors.
This script finds and fixes all affected files.

Run: python scripts/fix_validation_syntax_errors.py
"""

import os
import re
from pathlib import Path


def fix_file(filepath: Path) -> bool:
    """Fix validation placement in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: def fit(self, \n # Validate \n validate_... \n \n user_ids: ...
        # Should be: def fit(self, \n user_ids: ... ) -> ...: \n # Validate \n validate_...
        
        # Find problematic function definitions
        # Pattern: def NAME(self, <newline> # Validate <newline> validate_... <newline> <empty line> <params>
        pattern = r'(def\s+(fit|recommend)\s*\(\s*self\s*,)\s*\n\s*(#\s*Validate[^\n]*\n\s*validate_[^\n]+\n)\s*\n\s*([^\n]+:\s*[^\n]+,?\s*\n(?:[^\n]+[,)]?\s*\n)*[^\)]*\)\s*->[^:]+:)'
        
        def replacer(match):
            func_decl_start = match.group(1)  # "def fit(self,"
            func_name = match.group(2)  # "fit" or "recommend"
            validation_lines = match.group(3)  # "# Validate\n    validate_..."
            params_and_rest = match.group(4)  # "user_ids: ...) -> ...:"
            
            # Reconstruct properly: def fit(self, params) -> ...: \n validation
            return f"{func_decl_start}\n            {params_and_rest}\n        \n        {validation_lines}"
        
        # Try to fix with regex
        content = re.sub(pattern, replacer, content, flags=re.MULTILINE | re.DOTALL)
        
        # If regex didn't work, try manual approach
        if content == original_content:
            lines = content.split('\n')
            fixed_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # Check if this is a function definition with the problem
                if re.match(r'\s*def (fit|recommend)\s*\(\s*self\s*,\s*$', line):
                    # Check if next line starts with # Validate
                    if i + 1 < len(lines) and '# Validate' in lines[i + 1]:
                        # Found the problematic pattern
                        func_line = line
                        validation_start = i + 1
                        
                        # Collect validation lines
                        validation_lines = []
                        j = validation_start
                        while j < len(lines) and (lines[j].strip().startswith('#') or 
                                                 lines[j].strip().startswith('validate_') or
                                                 lines[j].strip() == ''):
                            validation_lines.append(lines[j])
                            j += 1
                            if lines[j-1].strip() and not lines[j-1].strip().startswith('#') and 'validate_' in lines[j-1]:
                                break
                        
                        # Find the actual parameters
                        param_lines = []
                        while j < len(lines):
                            param_lines.append(lines[j])
                            if ')' in lines[j] and ':' in lines[j]:
                                j += 1
                                break
                            j += 1
                        
                        # Reconstruct properly
                        fixed_lines.append(func_line)
                        fixed_lines.extend(param_lines)
                        fixed_lines.append('')  # Blank line
                        fixed_lines.extend(validation_lines)
                        
                        i = j
                        continue
                
                fixed_lines.append(line)
                i += 1
            
            content = '\n'.join(fixed_lines)
        
        # Only write if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False


def main():
    """Main function."""
    import subprocess
    
    # List of files known to have errors
    error_files = [
        'corerec/engines/deepfm.py',
        'corerec/engines/unionizedFilterEngine/sar.py',
        'corerec/engines/unionizedFilterEngine/nn_base/GateNet_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/DIN_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/FGCNN_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/AutoFI_base_test.py',
        'corerec/engines/unionizedFilterEngine/nn_base/DCN_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/FFM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/NFM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/ENSFM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/ESCMM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/ESMM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/Fibinet_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/DIFM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/AutoInt_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/FM_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/FLEN_base.py',
        'corerec/engines/unionizedFilterEngine/nn_base/nextitnet.py',
        'corerec/engines/contentFilterEngine/context_personalization/context_aware.py',
    ]
    
    base_dir = Path(__file__).parent.parent
    fixed_count = 0
    
    print(f"Fixing {len(error_files)} files with syntax errors...")
    print()
    
    for rel_path in error_files:
        filepath = base_dir / rel_path
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
                
                # Verify fix by compiling
                result = subprocess.run(['python', '-m', 'py_compile', str(filepath)],
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  ⚠ Still has syntax error: {filepath}")
                    print(f"     {result.stderr[:100]}")
        else:
            print(f"✗ File not found: {filepath}")
    
    print(f"\nFixed {fixed_count}/{len(error_files)} files")
    
    # Final verification
    print("\nVerifying all fixes...")
    all_good = True
    for rel_path in error_files:
        filepath = base_dir / rel_path
        if filepath.exists():
            result = subprocess.run(['python', '-m', 'py_compile', str(filepath)],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"✗ {rel_path}")
                all_good = False
    
    if all_good:
        print("✅ All syntax errors fixed!")
    else:
        print("\n⚠️  Some files still have syntax errors - manual fix needed")


if __name__ == '__main__':
    main()

