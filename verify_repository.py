"""
Repository Integrity Verification Script

Run this BEFORE pushing to GitHub to ensure:
1. All paths are relative (no hardcoded absolute paths)
2. Required directories exist
3. No sensitive data is tracked
4. Dependencies are installable
"""

import sys
import subprocess
from pathlib import Path
import re

print("=" * 80)
print("REPOSITORY INTEGRITY CHECK")
print("=" * 80)

# Test 1: Verify all Python scripts use relative paths
print("\n[Test 1] Checking for hardcoded absolute paths...")
py_files = list(Path(".").glob("*.py"))
absolute_path_pattern = re.compile(r'[C-Z]:\\|/Users/|/home/')

issues = []
for py_file in py_files:
    if py_file.name == "verify_repository.py":
        continue
    content = py_file.read_text(encoding='utf-8')
    if absolute_path_pattern.search(content):
        issues.append(f"  ❌ {py_file.name}: Contains absolute path")

if issues:
    for issue in issues:
        print(issue)
    print("  ⚠️  FIX: Replace with Path('relative/path')")
else:
    print("  ✅ All paths are relative")

# Test 2: Verify required directories
print("\n[Test 2] Checking directory structure...")
required_dirs = ["data", "outputs"]
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"  ✅ {dir_name}/ exists")
    else:
        print(f"  ⚠️  {dir_name}/ missing (will be created on first run)")

# Test 3: Verify .gitignore effectiveness
print("\n[Test 3] Checking .gitignore coverage...")
try:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True
    )
    tracked_files = result.stdout.strip().split('\n')
    
    sensitive_patterns = ['.pkl', '.csv', '.env', '__pycache__', 'data/', 'outputs/']
    sensitive_tracked = []
    
    for file in tracked_files:
        for pattern in sensitive_patterns:
            if pattern in file:
                sensitive_tracked.append(file)
    
    if sensitive_tracked:
        print("  ❌ Sensitive files tracked:")
        for file in sensitive_tracked:
            print(f"     - {file}")
        print("  ⚠️  FIX: Run 'git rm --cached <file>' and verify .gitignore")
    else:
        print("  ✅ No sensitive files tracked")
        
except subprocess.CalledProcessError:
    print("  ⚠️  Not a git repository (run 'git init' first)")

# Test 4: Verify required files exist
print("\n[Test 4] Checking essential repository files...")
required_files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "eda_information_gap.py",
    "preprocessing_stage2.py",
    "stage3_budget_aware_summarization.py",
    "stage4_gpu_optimized.py",
    "stage5_ground_truth_llm.py",
    "stage6_evaluation_metrics.py"
]

missing_files = []
for file in required_files:
    if Path(file).exists():
        print(f"  ✅ {file}")
    else:
        missing_files.append(file)
        print(f"  ❌ {file} MISSING")

if missing_files:
    print(f"\n  ⚠️  {len(missing_files)} required files missing!")
    sys.exit(1)

# Test 5: Verify Python syntax
print("\n[Test 5] Checking Python syntax...")
for py_file in py_files:
    if py_file.name == "verify_repository.py":
        continue
    try:
        compile(py_file.read_text(encoding='utf-8'), py_file.name, 'exec')
        print(f"  ✅ {py_file.name}: Valid syntax")
    except SyntaxError as e:
        print(f"  ❌ {py_file.name}: SyntaxError at line {e.lineno}")
        issues.append(f"Syntax error in {py_file.name}")

# Test 6: Verify imports can be resolved (basic check)
print("\n[Test 6] Checking critical imports...")
critical_imports = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("transformers", "AutoTokenizer"),
    ("sklearn", "TfidfVectorizer"),
    ("nltk", "sent_tokenize")
]

missing_packages = []
for package, module in critical_imports:
    try:
        if package == "sklearn":
            import sklearn
        elif package == "transformers":
            import transformers
        elif package == "pandas":
            import pandas
        elif package == "numpy":
            import numpy
        elif package == "nltk":
            import nltk
        print(f"  ✅ {package}")
    except ImportError:
        missing_packages.append(package)
        print(f"  ⚠️  {package} not installed (run: pip install {package})")

if missing_packages:
    print(f"\n  ℹ️  {len(missing_packages)} packages missing - install via:")
    print("     pip install -r requirements.txt")

# Test 7: Check repository size
print("\n[Test 7] Checking repository size...")
try:
    result = subprocess.run(
        ["git", "count-objects", "-vH"],
        capture_output=True,
        text=True,
        check=True
    )
    size_line = [line for line in result.stdout.split('\n') if 'size:' in line][0]
    size = size_line.split(':')[1].strip()
    
    print(f"  📊 Repository size: {size}")
    
    # Extract numeric value
    size_kb = float(size.split()[0])
    if size_kb > 1000:  # 1 MB threshold
        print("  ⚠️  Repository exceeds 1 MB - verify no large files committed")
    else:
        print("  ✅ Size is appropriate for code-only repository")
        
except (subprocess.CalledProcessError, IndexError):
    print("  ⚠️  Could not determine repository size")

# Final Summary
print("\n" + "=" * 80)
if issues:
    print("❌ VERIFICATION FAILED")
    print(f"   {len(issues)} issues found. Fix before pushing to GitHub.")
    for issue in issues:
        print(f"   - {issue}")
    sys.exit(1)
else:
    print("✅ ALL CHECKS PASSED")
    print("   Repository is ready for GitHub deployment!")
    print("\n   Next steps:")
    print("   1. Create GitHub repository at https://github.com/new")
    print("   2. Run: git remote add origin https://github.com/[username]/[repo].git")
    print("   3. Run: git branch -M main")
    print("   4. Run: git push -u origin main")
print("=" * 80)
