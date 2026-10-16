#!/usr/bin/env python3
"""
Accessibility Testing Script for FixOldImg

Tests:
1. Color contrast ratios
2. Touch target sizes
3. Focus states
4. Token usage (no hardcoded values)
5. Form labels and ARIA attributes

Usage:
    python scripts/accessibility-test.py [--verbose] [--fix]
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass


@dataclass
class AccessibilityIssue:
    file: str
    line: int
    category: str
    severity: str  # critical, warning, info
    message: str
    suggestion: str


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Calculate relative luminance per WCAG 2.1."""
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color1: str, color2: str) -> float:
    """Calculate contrast ratio between two hex colors."""
    lum1 = relative_luminance(hex_to_rgb(color1))
    lum2 = relative_luminance(hex_to_rgb(color2))
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def check_contrast_issues(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for color contrast issues."""
    issues = []
    lines = content.split('\n')

    # Common color pairs to check (foreground, background, context)
    color_pairs = [
        ('#f4f4f5', '#0f0f11', 'text on background'),
        ('#ffffff', '#ea580c', 'button text'),
        ('#86efac', '#14532d', 'success text'),
        ('#fecaca', '#7f1d1d', 'error text'),
    ]

    # Skip certain files (test scripts, etc.)
    skip_files = ['accessibility-test.py', 'test_']
    if any(skip in file_path for skip in skip_files):
        return issues

    for line_num, line in enumerate(lines, 1):
        for fg, bg, context in color_pairs:
            if fg in line and bg in line:
                ratio = contrast_ratio(fg, bg)
                if ratio < 4.5:
                    issues.append(AccessibilityIssue(
                        file=file_path,
                        line=line_num,
                        category='contrast',
                        severity='critical',
                        message=f'Insufficient contrast ratio {ratio:.2f}:1 for {context}',
                        suggestion='Use darker foreground or lighter background (need 4.5:1)'
                    ))

    return issues


def check_touch_targets(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for minimum touch target sizes (44x44px)."""
    issues = []
    lines = content.split('\n')

    # Skip scrollbar and decorative elements
    skip_patterns = ['scrollbar', 'webkit-scrollbar', 'keyframes']

    # Track context (previous lines)
    prev_lines = [''] * 5

    for line_num, line in enumerate(lines, 1):
        # Check if this line or nearby lines contain skip patterns
        context = ' '.join(prev_lines[-3:] + [line]).lower()
        if any(skip in context for skip in skip_patterns):
            prev_lines.append(line)
            continue

        # Check for height declarations
        height_match = re.search(r'height:\s*(\d+)px', line)
        if height_match:
            height = int(height_match.group(1))
            if height < 44:
                issues.append(AccessibilityIssue(
                    file=file_path,
                    line=line_num,
                    category='touch-target',
                    severity='warning',
                    message=f'Height {height}px is below 44px minimum',
                    suggestion='Increase height to 44px for touch accessibility'
                ))

        # Check for min-height
        min_height_match = re.search(r'min-height:\s*(\d+)px', line)
        if min_height_match:
            height = int(min_height_match.group(1))
            if height < 44:
                issues.append(AccessibilityIssue(
                    file=file_path,
                    line=line_num,
                    category='touch-target',
                    severity='warning',
                    message=f'min-height {height}px is below 44px minimum',
                    suggestion='Increase min-height to 44px for touch accessibility'
                ))

        prev_lines.append(line)

    return issues


def check_focus_states(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for focus state definitions."""
    issues = []

    # Check if focus-visible is defined
    if ':focus-visible' not in content and ':focus' not in content:
        issues.append(AccessibilityIssue(
            file=file_path,
            line=0,
            category='focus',
            severity='warning',
            message='No focus state defined',
            suggestion='Add :focus-visible styles for keyboard accessibility'
        ))

    return issues


def check_reduced_motion(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for reduced motion support."""
    issues = []

    if 'prefers-reduced-motion' not in content:
        issues.append(AccessibilityIssue(
            file=file_path,
            line=0,
            category='motion',
            severity='info',
            message='No reduced motion support',
            suggestion='Add @media (prefers-reduced-motion: reduce) query'
        ))

    return issues


def check_token_usage(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for hardcoded values that should use tokens."""
    issues = []

    # Hardcoded colors (should use tokens)
    hardcoded_colors = re.findall(r'#[0-9a-fA-F]{3,8}', content)
    if hardcoded_colors:
        # Filter out common exceptions
        exceptions = {'#000', '#fff', '#000000', '#ffffff'}
        real_hardcoded = [c for c in hardcoded_colors if c.lower() not in exceptions]

        if real_hardcoded:
            issues.append(AccessibilityIssue(
                file=file_path,
                line=0,
                category='tokens',
                severity='info',
                message=f'Found {len(real_hardcoded)} hardcoded color values',
                suggestion='Consider using CSS variables for maintainability'
            ))

    # Hardcoded spacing
    hardcoded_spacing = re.findall(r'(?<!\w)(\d+)px', content)
    if hardcoded_spacing:
        # Filter out common exceptions (0px, 1px, 2px borders)
        real_hardcoded = [s for s in hardcoded_spacing if int(s) > 2]
        if real_hardcoded:
            issues.append(AccessibilityIssue(
                file=file_path,
                line=0,
                category='tokens',
                severity='info',
                message=f'Found {len(real_hardcoded)} hardcoded spacing values',
                suggestion='Consider using spacing tokens (--space-*)'
            ))

    return issues


def check_form_labels(file_path: str, content: str) -> list[AccessibilityIssue]:
    """Check for form labels and ARIA attributes."""
    issues = []

    # Check for inputs without labels
    input_pattern = re.compile(r'<input[^>]*>', re.IGNORECASE)
    label_pattern = re.compile(r'<label[^>]*for=["\']([^"\']+)["\']', re.IGNORECASE)

    inputs = input_pattern.findall(content)
    labels = label_pattern.findall(content)

    for input_match in inputs:
        id_match = re.search(r'id=["\']([^"\']+)["\']', input_match)
        if id_match:
            input_id = id_match.group(1)
            # A control with no <label> needs an explicit aria-label of its own.
            if input_id not in labels and (
                'aria-label' not in input_match and 'aria-labelledby' not in input_match
            ):
                issues.append(AccessibilityIssue(
                    file=file_path,
                    line=0,
                    category='forms',
                    severity='warning',
                    message=f'Input #{input_id} has no associated label',
                    suggestion='Add <label for="{input_id}"> or aria-label',
                ))

    return issues


def scan_file(file_path: str, verbose: bool = False) -> list[AccessibilityIssue]:
    """Scan a single file for accessibility issues."""
    issues = []

    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return issues

    # Determine file type
    is_css = file_path.endswith('.css')
    is_html = file_path.endswith('.html')

    # Run appropriate checks based on file type
    issues.extend(check_contrast_issues(file_path, content))
    issues.extend(check_touch_targets(file_path, content))

    # Only check CSS-specific issues for CSS/HTML files
    if is_css or is_html:
        issues.extend(check_focus_states(file_path, content))
        issues.extend(check_reduced_motion(file_path, content))

    # Only check token usage for CSS files (not design-tokens.css itself)
    if is_css and 'design-tokens.css' not in file_path:
        issues.extend(check_token_usage(file_path, content))

    # Only check form labels for HTML files
    if is_html:
        issues.extend(check_form_labels(file_path, content))

    return issues


def scan_directory(directory: str, verbose: bool = False) -> list[AccessibilityIssue]:
    """Scan all HTML, CSS, and Python files in a directory."""
    all_issues = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']

        for file in files:
            if file.endswith(('.html', '.css', '.py')):
                file_path = os.path.join(root, file)
                issues = scan_file(file_path, verbose)
                all_issues.extend(issues)

                if verbose and issues:
                    print(f"\n[FILE] {file_path}: {len(issues)} issues")

    return all_issues


def print_report(issues: list[AccessibilityIssue], verbose: bool = False):
    """Print a formatted accessibility report."""
    if not issues:
        print("\n[OK] No accessibility issues found!")
        return

    # Group by severity
    critical = [i for i in issues if i.severity == 'critical']
    warning = [i for i in issues if i.severity == 'warning']
    info = [i for i in issues if i.severity == 'info']

    print("\n[REPORT] Accessibility Report")
    print("=" * 60)

    if critical:
        print(f"\n[CRITICAL] ({len(critical)} issues)")
        print("-" * 60)
        for issue in critical:
            print(f"  {issue.file}:{issue.line}")
            print(f"    {issue.message}")
            print(f"    [FIX] {issue.suggestion}")
            print()

    if warning:
        print(f"\n[WARNING] ({len(warning)} issues)")
        print("-" * 60)
        for issue in warning:
            print(f"  {issue.file}:{issue.line}")
            print(f"    {issue.message}")
            print(f"    [FIX] {issue.suggestion}")
            print()

    if info:
        print(f"\n[INFO] ({len(info)} issues)")
        print("-" * 60)
        for issue in info:
            print(f"  {issue.file}:{issue.line}")
            print(f"    {issue.message}")
            print(f"    [FIX] {issue.suggestion}")
            print()

    print("=" * 60)
    print(f"Total: {len(issues)} issues ({len(critical)} critical, {len(warning)} warnings, {len(info)} info)")


def main():
    # Set stdout encoding to UTF-8 for Windows compatibility
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description='Accessibility testing for FixOldImg')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--fix', action='store_true', help='Attempt to auto-fix issues')
    parser.add_argument('--directory', '-d', default='.', help='Directory to scan')

    args = parser.parse_args()

    print("[a] Accessibility Testing Tool")
    print("=" * 60)

    issues = scan_directory(args.directory, args.verbose)
    print_report(issues, args.verbose)

    # Exit with error code if critical issues found
    critical_count = len([i for i in issues if i.severity == 'critical'])
    sys.exit(1 if critical_count > 0 else 0)


if __name__ == '__main__':
    main()
