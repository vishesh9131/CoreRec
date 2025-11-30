#!/usr/bin/env python3
"""Remove badly inserted validation lines."""

files = [
    "corerec/engines/unionizedFilterEngine/nn_base/GateNet_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DIN_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/FGCNN_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/AutoFI_base_test.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DCN_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/FFM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/NFM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/ENSFM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/ESCMM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/ESMM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/Fibinet_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DIFM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/AutoInt_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/FM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/FLEN_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/nextitnet.py",
    "corerec/engines/contentFilterEngine/context_personalization/context_aware.py",
]

for fpath in files:
    try:
        with open(fpath, "r") as f:
            lines = f.readlines()

        # Remove lines with validation that appear before function signature
        # closes
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # If we see "# Validate" before a closing paren with ->, skip it
            # and next validate line
            if "# Validate" in line:
                # Check if we're in a function signature (look ahead for
                # closing paren)
                in_signature = False
                for j in range(i - 5, min(i + 10, len(lines))):
                    if j >= 0 and "def " in lines[j] and "(" in lines[j]:
                        # Check if signature is closed
                        closed = False
                        for k in range(j, min(j + 15, len(lines))):
                            if ")" in lines[k] and "->" in lines[k] and ":" in lines[k]:
                                closed = True
                                break
                        if not closed and i < k:
                            in_signature = True
                            break

                if in_signature:
                    print(
                        f"  Removing bad validation from {fpath} line {
                            i + 1}")
                    i += 1  # Skip # Validate line
                    # Skip validate_xxx line
                    while i < len(lines) and (
                            "validate_" in lines[i] or lines[i].strip() == ""):
                        i += 1
                    continue

            new_lines.append(line)
            i += 1

        with open(fpath, "w") as f:
            f.writelines(new_lines)
        print(f"✓ {fpath}")

    except Exception as e:
        print(f"✗ {fpath}: {e}")

print("Done!")
